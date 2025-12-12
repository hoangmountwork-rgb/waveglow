import torch
import numpy as np
import soundfile as sf
import torchaudio
from glow import WaveGlow  # model class

# ---------------------------------------
#  Load WaveGlow (pretrained NVIDIA)
# ---------------------------------------
def load_waveglow(path, device):
    checkpoint = torch.load(path, map_location=device)

    # 1️⃣ Tạo model trống giống config khi train
    waveglow = WaveGlow(
        n_mel_channels=80,
        n_flows=8,
        n_group=8,
        n_early_every=4,
        n_early_size=2,
        WN_config={
            "n_layers": 6,
            "n_channels": 192,
            "kernel_size": 3
        }
    ).to(device)

    # 2️⃣ Load state dict
    if "model" in checkpoint:
        waveglow.load_state_dict(checkpoint["model"])
    else:
        waveglow.load_state_dict(checkpoint)

    # 3️⃣ Remove weight norm
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.eval()

    print("✅ Loaded pretrained WaveGlow successfully.")
    return waveglow

# ---------------------------------------
#  Generate mel
# ---------------------------------------
def wav_to_mel(wav_path, device):
    wav, sr = torchaudio.load(wav_path)

    # Resample to 22050Hz
    if sr != 22050:
        wav = torchaudio.functional.resample(wav, sr, 22050)

    wav = wav.mean(dim=0, keepdim=True)   # mono
    wav = wav.to(device)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        f_min=0,
        f_max=8000
    ).to(device)

    mel = mel_transform(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))

    return mel

# ---------------------------------------
#  Inference
# ---------------------------------------
def infer_waveglow(waveglow, mel, device, sigma=0.6):
    with torch.no_grad():
        audio = waveglow.infer(mel, sigma=sigma)

    audio = audio.squeeze().cpu().numpy()
    audio = audio / np.max(np.abs(audio))  # normalize
    return audio

# ---------------------------------------
#  Main
# ---------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Path pretrained waveglow
    waveglow_path = r"E:\waveglow\waveglow\checkpoints\waveglow_15000.pt"
    input_wav = "example.wav"  # file ví dụ

    # Load model
    waveglow = load_waveglow(waveglow_path, device)

    # Convert wav → mel
    mel = wav_to_mel(input_wav, device)

    # Infer audio
    print("Generating audio...")
    audio = infer_waveglow(waveglow, mel, device, sigma=0.6)

    # Save result
    sf.write("output.wav", audio, 22050)
    print("✅ Done! Saved as output.wav")

if __name__ == "__main__":
    main()
