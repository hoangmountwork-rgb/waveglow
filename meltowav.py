import os
import json
import torch
import numpy as np
from glow import WaveGlow
import soundfile as sf
from scipy.signal import butter, filtfilt

def soft_denoise(audio, threshold=0.02):
    """
    Giảm nhẹ các vùng tín hiệu nhỏ (< threshold), giữ vùng mạnh.
    Không dùng noise ngẫu nhiên.
    """
    audio = audio.copy()
    mask = np.abs(audio) < threshold
    audio[mask] *= 0.5  # giảm 50% vùng yếu
    return audio

def lowpass_filter(audio, fs, cutoff=6000, order=4):
    """
    Low-pass filter nhẹ, cutoff ~6000Hz, giữ voice rõ nhưng loại bỏ high freq hiss.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_audio = filtfilt(b, a, audio)
    return filtered_audio

def normalize_audio(audio):
    """
    Normalize amplitude về [-0.99, 0.99] để tránh clipping.
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = 0.99 * audio / max_val
    return audio

def main():
    # --- Cấu hình ---
    config_path = "config.json"
    mel_path = "mel/VIVOSSPK01_R002.npy"
    checkpoint_path = "checkpoints/waveglow_256channels.pt"
    output_dir = "generated_wav"

    # Load config
    config = json.load(open(config_path))
    data_cfg = config["data_config"]
    wg_cfg = config["waveglow_config"]

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WaveGlow(**wg_cfg).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"✅ Loaded model from {checkpoint_path}")

    # Load mel
    mel = np.load(mel_path)
    mel = torch.from_numpy(mel).unsqueeze(0).float().to(device)
    print(f"✅ Loaded mel from {mel_path} (shape: {mel.shape})")

    # --- Infer ---
    sigma = 1.0  # Sigma vừa đủ, không quá nhỏ để tránh rè
    with torch.no_grad():
        generated_audio = model.infer(mel, sigma=sigma)
    generated_audio = generated_audio.cpu().squeeze(0).float()
    generated_np = generated_audio.numpy()

    # --- Denoise nhẹ ---
    generated_np = soft_denoise(generated_np, threshold=0.02)

    # --- Low-pass filter nhẹ ---
    fs = data_cfg["sampling_rate"]
    filtered_audio = lowpass_filter(generated_np, fs, cutoff=6000, order=4)

    # --- Normalize để âm thanh đủ lớn ---
    filtered_audio = normalize_audio(filtered_audio)

    # --- Lưu file ---
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(mel_path).replace('.npy', f'_clear_sigma{sigma}_cutoff6000.wav')
    output_wav = os.path.join(output_dir, filename)
    sf.write(output_wav, filtered_audio, fs)
    print(f"✅ Generated '{output_wav}' – voice rõ, ít noise, nghe tự nhiên")

if __name__ == "__main__":
    main()
