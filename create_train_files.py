import os

# Thư mục chứa file .wav
train_dir = r"E:\waveglow\waveglow\vivos\train\waves_22050"

# File output
output_file = "train_files.txt"

# Kiểm tra thư mục tồn tại
if not os.path.exists(train_dir):
    print(f"❌ Thư mục không tồn tại: {train_dir}")
else:
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for root, _, files in os.walk(train_dir):
            for file in sorted(files):
                if file.lower().endswith(".wav"):
                    full_path = os.path.join(root, file)
                    f.write(full_path + "\n")
                    count += 1
    print(f"✅ {output_file} created with {count} .wav files")
