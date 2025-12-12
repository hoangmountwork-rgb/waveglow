import os
import torch
import numpy as np
import soundfile as sf
from glow import WaveGlow
from collections import OrderedDict

# ---------------- CONFIG ----------------
mel_path = "mel/VIVOSSPK01_R001.npy"  # Mel dài của đoạn văn
checkpoint_path = "checkpoints/waveglow_2000.pt"
output_wav = "generated_long.wav"
sigma = 1.0  # độ mạnh tiếng

segment_frames = 64   # Số frame mel mỗi segment (~0.74s với hop_length=256, segment_length=16384)
crossfade_samples = 256  # Số samples để crossfade khi nối

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODEL ----------------
wg_cfg = {
    "n_mel_channels": 80,
    "n_flows": 8,
    "n_group": 8,
    "n_early_every": 4,
    "n_early_size": 2,
    "WN_config": {
        "n_layers": 6,
        "n_channels": 192,
        "kernel_size": 3
    }
}

model = WaveGlow(**wg_cfg).to(device).eval()
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint["model"]
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith("module."):
        k = k.replace("module.", "")
    new_state_dict[k] = v
model.load_state_dict(new_state_dict)
print("✅ Model loaded")

# ---------------- LOAD MEL ----------------
mel = np.load(mel_path).astype(np.float32)  # shape [80, T]
mel = torch.from_numpy(mel).unsqueeze(0).to(device)  # [1, 80, T]
n_frames = mel.shape[2]
print(f"Mel shape: {mel.shape} → {n_frames} frames")

# ---------------- INFER SEGMENT ----------------
audio_segments = []
start = 0
while start < n_frames:
    end = min(start + segment_frames, n_frames)
    mel_segment = mel[:, :, start:end]
    with torch.no_grad():
        audio_segment = model.infer(mel_segment, sigma=sigma)
    audio_segment = audio_segment.cpu().numpy().squeeze()
    audio_segments.append(audio_segment)
    start += segment_frames

# ---------------- NỐI SEGMENT ----------------
def crossfade_concat(a, b, overlap):
    if overlap == 0:
        return np.concatenate([a, b])
    fade = np.linspace(0, 1, overlap)
    a[-overlap:] = a[-overlap:] * (1 - fade)
    b[:overlap] = b[:overlap] * fade
    return np.concatenate([a, b])

audio_final = audio_segments[0]
for seg in audio_segments[1:]:
    audio_final = crossfade_concat(audio_final, seg, crossfade_samples)

# ---------------- SAVE ----------------
fs = 22050  # sampling_rate, khớp config train
sf.write(output_wav, audio_final, fs)
print(f"✅ Generated audio dài: {output_wav} (voice rõ, không rè)")
