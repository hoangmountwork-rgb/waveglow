import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import MelSpectrogram
from glow import WaveGlow, WaveGlowLoss
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class VivosDataset(Dataset):
    def __init__(self, filelist, config):
        with open(filelist, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        self.wav_files = [line.split('|')[0] for line in lines]

        self.sampling_rate = config["sampling_rate"]
        self.segment_length = config["segment_length"]

        self.mel_transform = MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=config["filter_length"],
            hop_length=config["hop_length"],
            win_length=config["win_length"],
            f_min=config["mel_fmin"],
            f_max=config["mel_fmax"],
            n_mels=80
        )

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]
        audio, sr = torchaudio.load(wav_path)

        audio = torch.mean(audio, dim=0, keepdim=True)

        if sr != self.sampling_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sampling_rate)

        if audio.size(1) > self.segment_length:
            max_start = audio.size(1) - self.segment_length
            start = torch.randint(0, max_start, (1,)).item()
            audio = audio[:, start:start + self.segment_length]
        else:
            pad_len = self.segment_length - audio.size(1)
            audio = nn.functional.pad(audio, (0, pad_len), "constant")

        mel = self.mel_transform(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        # Fix cho old PyTorch: thay nan_to_num bằng where + clamp
        mel = torch.where(torch.isnan(mel), torch.zeros_like(mel), mel)
        mel = torch.clamp(mel, min=-1e5, max=1e5)
        audio = torch.where(torch.isnan(audio), torch.zeros_like(audio), audio)
        audio = torch.clamp(audio, min=-1e5, max=1e5)

        return mel.squeeze(0), audio.squeeze(0)


def save_checkpoint(model, opt, step, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"waveglow_{step}.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step
    }, path)
    print(f"Saved: {path}")


def load_checkpoint(model, opt, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        print(f"Loaded checkpoint at step {step}")
        return step
    return 0


def main():
    config = json.load(open("config.json"))
    train_cfg = config["train_config"]
    data_cfg = config["data_config"]
    wg_cfg = config["waveglow_config"]

    torch.manual_seed(train_cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg["seed"])
    random.seed(train_cfg["seed"])

    output_dir = train_cfg["output_directory"]
    os.makedirs(output_dir, exist_ok=True)

    dataset = VivosDataset(data_cfg["training_files"], data_cfg)
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WaveGlow(**wg_cfg).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"], betas=(0.8, 0.999))
    criterion = WaveGlowLoss(sigma=train_cfg["sigma"])

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2, eta_min=1e-6)

    step = load_checkpoint(model, optimizer, train_cfg["checkpoint_path"])

    use_fp16 = train_cfg["fp16_run"] and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    total_steps = train_cfg["epochs"]
    iters_per_checkpoint = train_cfg["iters_per_checkpoint"]

    print(f"Starting training from step {step}/{total_steps}, batch_size={train_cfg['batch_size']}, fp16={use_fp16}, device={device}")

    pbar = tqdm(total=total_steps - step, initial=step, desc="Training")
    epoch_iter = iter(loader)
    for current_step in range(step + 1, total_steps + 1):
        try:
            mel, audio = next(epoch_iter)
        except StopIteration:
            epoch_iter = iter(loader)
            mel, audio = next(epoch_iter)

        mel = mel.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_fp16:
            with torch.cuda.amp.autocast():
                z, log_s_list, log_det_W_list = model((mel, audio))
                loss = criterion((z, log_s_list, log_det_W_list))
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Skip bad batch at step {current_step}")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            z, log_s_list, log_det_W_list = model((mel, audio))
            loss = criterion((z, log_s_list, log_det_W_list))
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Skip bad batch at step {current_step}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        scheduler.step()

        if current_step % 100 == 0:
            grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            print(f"Grad norm: {grad_norm:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        pbar.update(1)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        if current_step % 50 == 0:
            print(f"[Step {current_step}/{total_steps}] Loss = {loss.item():.4f}")

        if current_step % iters_per_checkpoint == 0:
            save_checkpoint(model, optimizer, current_step, output_dir)

    save_checkpoint(model, optimizer, total_steps, output_dir)
    pbar.close()
    print("Training completed!")


if __name__ == "__main__":
    main()