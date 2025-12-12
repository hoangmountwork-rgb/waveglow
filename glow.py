import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def fused_add_tanh_sigmoid_multiply(a, b, n_channels):
    in_act = a + b
    c = n_channels
    t = torch.tanh(in_act[:, :c, :])
    s = torch.sigmoid(in_act[:, c:, :])
    return t * s


class Invertible1x1Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        W = torch.qr(torch.randn(channels, channels))[0]
        self.weight = nn.Parameter(W)

    def forward(self, x):
        B, C, T = x.size()
        W = self.weight
        z = torch.matmul(W.unsqueeze(0), x)
        log_det = T * torch.slogdet(W)[1]
        return z, log_det

    def inverse(self, z):
        W_inv = torch.inverse(self.weight)
        x = torch.matmul(W_inv.unsqueeze(0), z)
        return x


class WN(nn.Module):
    def __init__(self, n_audio_channels, cond_channels, n_layers=8, n_channels=256, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.n_layers = n_layers
        self.n_audio = n_audio_channels
        self.n_channels = n_channels

        self.start = nn.Conv1d(n_audio_channels, n_channels, kernel_size=1)

        self.in_layers = nn.ModuleList()
        self.cond_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.in_layers.append(nn.Conv1d(n_channels, n_channels * 2, kernel_size, padding=padding))
            self.cond_layers.append(nn.Conv1d(cond_channels, n_channels * 2, kernel_size=1))
            self.res_skip_layers.append(nn.Conv1d(n_channels, n_channels, kernel_size=1))

        self.end = nn.Conv1d(n_channels, 2 * n_audio_channels, kernel_size=1)

    def forward(self, x):
        audio, cond = x
        h = self.start(audio)

        for in_conv, cond_conv, res_conv in zip(self.in_layers, self.cond_layers, self.res_skip_layers):
            a = in_conv(h)
            b = cond_conv(cond)
            n = self.n_channels
            t_act = torch.tanh((a + b)[:, :n, :])
            s_act = torch.sigmoid((a + b)[:, n:, :])
            acts = t_act * s_act
            h = h + res_conv(acts)

        out = self.end(h)
        return out


class WaveGlow(nn.Module):
    def __init__(self, n_mel_channels=80, n_flows=12, n_group=8, n_early_every=4, n_early_size=2,
                 WN_config=None):
        super().__init__()
        assert WN_config is not None

        self.n_mel_channels = n_mel_channels
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size

        self.upsample = nn.ConvTranspose1d(n_mel_channels, n_mel_channels,
                                           kernel_size=1024, stride=256, padding=(1024 - 256) // 2)

        self.convinv = nn.ModuleList()
        self.WN = nn.ModuleList()

        n_audio_channels = n_group // 2
        cond_channels = n_mel_channels

        for k in range(n_flows):
            self.convinv.append(Invertible1x1Conv(n_group))
            self.WN.append(WN(n_audio_channels, cond_channels,
                              n_layers=WN_config["n_layers"],
                              n_channels=WN_config["n_channels"],
                              kernel_size=WN_config.get("kernel_size", 3)
                              ))

    def forward(self, x):
        mel, audio = x
        B = mel.size(0)

        mel = self.upsample(mel)
        T_audio = audio.size(1)
        if mel.size(2) >= T_audio:
            mel = mel[:, :, :T_audio]
        else:
            mel = F.pad(mel, (0, T_audio - mel.size(2)), "constant")

        B, T = audio.size()
        assert T % self.n_group == 0, f"audio length {T} must be divisible by n_group {self.n_group}"

        audio = audio.view(B, self.n_group, T // self.n_group)

        z = audio
        log_s_list = []
        log_det_list = []

        for k in range(self.n_flows):
            z, log_det_W = self.convinv[k](z)
            log_det_list.append(log_det_W)

            audio_0 = z[:, : (self.n_group // 2), :]
            audio_1 = z[:, (self.n_group // 2):, :]

            Tg = z.size(2)
            mel_reshaped = mel.view(B, self.n_mel_channels, Tg, self.n_group)
            mel_group = mel_reshaped.mean(dim=3)

            h = self.WN[k]((audio_0, mel_group))
            m = h[:, : (self.n_group // 2), :]
            logs = h[:, (self.n_group // 2):, :]
            logs = torch.clamp(logs, min=-10.0, max=10.0)  # Clamp chống exp overflow

            audio_1 = (audio_1 - m) * torch.exp(-logs)

            z = torch.cat([audio_0, audio_1], dim=1)

            log_s_list.append(logs)

        return z, log_s_list, log_det_list


    def infer(self, mel, sigma=1.0):
        with torch.no_grad():
            mel = self.upsample(mel)
            T_audio = mel.size(2)
            Tg = T_audio // self.n_group
            z = torch.randn(1, self.n_group, Tg).to(mel.device) * sigma

            for k in reversed(range(self.n_flows)):
                audio_0 = z[:, : (self.n_group // 2), :]
                audio_1 = z[:, (self.n_group // 2):, :]

                mel_reshaped = mel.view(1, self.n_mel_channels, Tg, self.n_group)
                mel_group = mel_reshaped.mean(dim=3)

                h = self.WN[k]((audio_0, mel_group))
                m = h[:, : (self.n_group // 2), :]
                logs = h[:, (self.n_group // 2):, :]
                logs = torch.clamp(logs, min=-10.0, max=10.0)

                audio_1 = audio_1 * torch.exp(logs) + m
                z = torch.cat([audio_0, audio_1], dim=1)

                z = self.convinv[k].inverse(z)

            out = z.view(1, -1)
            return out


class WaveGlowLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        z, log_s_list, log_det_W_list = x
        # Fix: Dùng torch.stack.sum() thay python sum để ổn định
        log_det_W = torch.stack(log_det_W_list).sum()
        log_det_s = torch.stack([log_s.sum() for log_s in log_s_list]).sum()
        log_det = log_det_W + log_det_s
        log_det = torch.clamp(log_det, min=-1e6, max=1e6)  # Clamp total

        z_clamped = torch.clamp(z, min=-50.0, max=50.0)  # Clamp z
        log_2pi_sigma = 0.5 * math.log(2 * math.pi * self.sigma ** 2)
        const_term = log_2pi_sigma * z.numel()
        z_term = 0.5 * ((z_clamped / self.sigma) ** 2).sum()
        nll = const_term + z_term - log_det

        total_samples = z.size(0) * z.size(1) * z.size(2)
        loss = nll / total_samples

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(1e4, device=z.device, requires_grad=True)
        return loss