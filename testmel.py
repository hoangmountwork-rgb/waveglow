import librosa
import numpy as np

audio_path = "audio/VIVOSSPK01_R001.wav"  # file audio gốc
audio, sr = librosa.load(audio_path, sr=None)  # giữ sampling_rate gốc

print("Audio length (samples):", len(audio))
print("Audio duration (s):", len(audio)/sr)
