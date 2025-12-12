WaveGlow Vietnamese TTS
Repository này chứa hệ thống Text-to-Speech (TTS) bằng WaveGlow cho tiếng Việt. Cho phép bạn train model của riêng mình hoặc tạo audio từ checkpoint đã có sẵn.
Cấu trúc project

waveglow/
├── checkpoints/          # Checkpoint model (pretrained hoặc của bạn)
├── mel/                  # Mel spectrogram cho inference
├── vivos/                # thư mục chứa các file wav
├── train_files.txt       # Danh sách file .wav để train
├── config.json            # Cấu hình train & model
├── glow.py               # Implementation của WaveGlow
├── meltowav.py           # Script inference đơn giản
├── inference.py          # Script inference theo chuẩn NVIDIA
└── README.md

Yêu cầu
- Python 3.8+
- PyTorch 2.x (GPU + CUDA nếu có)
- numpy, soundfile, scipy

Cài đặt thư viện:
pip install torch numpy soundfile scipy
Tùy chọn: sử dụng conda environment:

conda create -n waveglow_tts python=3.8
conda activate waveglow_tts
pip install torch numpy soundfile scipy

Huấn luyện (Training)
Chuẩn bị dữ liệu:
- link checkpoints và data vivos https://drive.google.com/drive/folders/1C2BXouxKQh5ishqY0ms1HEDnu-Ta2Rd2?dmr=1&ec=wgc-drive-hero-goto (giải nén vào thư mục checkpoints)
- Sắp xếp các file .wav vào một thư mục.
- Tạo train_files.txt chứa đường dẫn tất cả các file .wav.
- Cấu hình các tham số trong config.json:

Bắt đầu huấn luyện:
python train.py --config config.json

Lưu ý:
- Với laptop Intel 11500H + RTX 3050 4GB:
  - Batch size = 4 là ổn.
  - FP16 giúp giảm bộ nhớ GPU.
- Loss có thể không giảm nhanh trong vài trăm bước đầu.

Inference (Tạo audio)
Sử dụng Mel Spectrogram:
python meltowav.py
Mẹo & Lưu ý

- Test ban đầu với segment length nhỏ và batch size nhỏ để tránh out-of-memory.
- Sử dụng checkpoint từng bước (iters_per_checkpoint) để lưu tiến trình training.
- Normalize audio sau khi infer để tránh clipping:
  audio = 0.99 * audio / np.max(np.abs(audio))
- Nếu gặp lỗi weightnorm, hãy loại bỏ hoặc dùng checkpoint tương thích.

Tài liệu tham khảo
- NVIDIA WaveGlow: https://github.com/NVIDIA/waveglow
- Dữ liệu TTS tiếng Việt: https://www.kaggle.com/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr
License
Project này dành cho nghiên cứu và sử dụng cá nhân.
Sử dụng thương mại cần tuân thủ license của NVIDIA WaveGlow.
