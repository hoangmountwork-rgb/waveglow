import librosa
import soundfile as sf
import os

input_root = r"E:\waveglow\vivos\train\waves"
output_root = r"E:\waveglow\waveglow\vivos\train\waves_22050"
os.makedirs(output_root, exist_ok=True)

count = 0

for speaker in os.listdir(input_root):
    spk_in = os.path.join(input_root, speaker)
    spk_out = os.path.join(output_root, speaker)
    os.makedirs(spk_out, exist_ok=True)

    for file in os.listdir(spk_in):
        if file.endswith(".wav"):
            in_path = os.path.join(spk_in, file)
            out_path = os.path.join(spk_out, file)

            y, sr = librosa.load(in_path, sr=22050)
            sf.write(out_path, y, 22050)

            count += 1
            print("✔", out_path)

print(f"\n🎉 Done! Resampled {count} files.")
