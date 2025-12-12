import os
import json
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from glow import WaveGlow
import soundfile as sf  # pip install soundfile nếu chưa có


def main():
    config_path = "config.json"
    checkpoint_path = "checkpoints/waveglow_15000.pt"  # Thay checkpoint anh muốn

    # Load config
    config = json.load(open(config_path))
    data_cfg = config["data_config"]
    wg_cfg = config["waveglow_config"]

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WaveGlow(**wg_cfg).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded model from {checkpoint_path}")

    # Lấy sample đầu từ train_files.txt
    with open(data_cfg["training_files"], "r") as f:
        first_line = f.readline().strip()
        if '|' in first_line:
            first_wav = first_line.split('|')[0]
        else:
            first_wav = first_line

    # Load audio sample
    audio, sr = torchaudio.load(first_wav)
    audio = torch.mean(audio, dim=0, keepdim=True)  # Mono [1, T]

    if sr != data_cfg["sampling_rate"]:
        audio = torchaudio.functional.resample(audio, sr, data_cfg["sampling_rate"])

    # Trim nếu dài hơn segment
    segment_length = data_cfg["segment_length"]
    if audio.size(1) > segment_length:
        audio = audio[:, :segment_length]

    # Compute mel (shape [1, 80, T_mel])
    mel_transform = MelSpectrogram(
        sample_rate=data_cfg["sampling_rate"],
        n_fft=data_cfg["filter_length"],
        hop_length=data_cfg["hop_length"],
        win_length=data_cfg["win_length"],
        f_min=data_cfg["mel_fmin"],
        f_max=data_cfg["mel_fmax"],
        n_mels=80
    )
    mel = mel_transform(audio)
    mel = torch.log(torch.clamp(mel, min=1e-5)).to(device)  # [1, 80, T_mel]

    # Infer với sigma thấp để giảm noise "sôi"
    sigma = 0.4  # FIX: Giảm từ 1.0 xuống 0.6 (thử 0.4-0.8 tùy anh thích êm hay random)
    with torch.no_grad():
        generated_audio = model.infer(mel, sigma=sigma)
    generated_audio = generated_audio.cpu().squeeze(0)

    # Normalize FIX: Scale đúng [-1,1] + clip để tránh distortion/rè
    generated_audio = generated_audio.float() / 32768.0  # 16-bit signed
    generated_audio = torch.clamp(generated_audio, min=-1.0, max=1.0)

    # Save WAV
    output_wav = f"generated_sample_sigma_{sigma}.wav"  # Tên file có sigma để phân biệt
    sf.write(output_wav, generated_audio.numpy(), data_cfg["sampling_rate"])
    print(f"✅ Generated '{output_wav}' với sigma={sigma} – nghe thử nhé anh! (Sample từ: {first_wav})")
    print("💡 Nếu vẫn rè: Thử sigma=0.4 (êm hơn) hoặc train thêm 5k steps.")


if __name__ == "__main__":
    main()