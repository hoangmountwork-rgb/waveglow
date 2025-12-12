import os
import json
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
import numpy as np
from tqdm import tqdm


def main():
    config_path = "config.json"
    filelist_path = "train_files.txt"  # Hoặc anh chỉnh nếu khác
    mel_dir = "mel"  # Folder lưu .npy

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    data_cfg = config["data_config"]

    # Tạo folder mel nếu chưa có
    os.makedirs(mel_dir, exist_ok=True)

    # Đọc list files
    with open(filelist_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    wav_files = [line.split('|')[0] for line in lines]  # chỉ lấy path

    # Mel transform
    mel_transform = MelSpectrogram(
        sample_rate=data_cfg["sampling_rate"],
        n_fft=data_cfg["filter_length"],
        hop_length=data_cfg["hop_length"],
        win_length=data_cfg["win_length"],
        f_min=data_cfg["mel_fmin"],
        f_max=data_cfg["mel_fmax"],
        n_mels=80,
        power=1.0  # magnitude spectrogram
    )

    print(f"🚀 Extracting mels for {len(wav_files)} files... (save to '{mel_dir}/')")

    for wav_path in tqdm(wav_files):
        # Load audio
        audio, sr = torchaudio.load(wav_path)  # [channels, samples]
        # Chuyển về mono
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Resample nếu sample rate khác
        if sr != data_cfg["sampling_rate"]:
            audio = torchaudio.functional.resample(audio, sr, data_cfg["sampling_rate"])

        # Tạo mel
        mel = mel_transform(audio)                # [1, 80, T]
        mel = torch.log(torch.clamp(mel, min=1e-5))  # log-mel
        mel = mel.squeeze(0).numpy()              # [80, T]

        # Save .npy
        filename = os.path.basename(wav_path).replace(".wav", ".npy")
        mel_path = os.path.join(mel_dir, filename)
        np.save(mel_path, mel)

    print("✅ All mels extracted! Check folder 'mel/' nhé anh.")


if __name__ == "__main__":
    main()
