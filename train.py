import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from model import WaveGlow
from mel import MelSpectrogram
from dataset import AudioMelDataset


def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)


def save_checkpoint(model, optimizer, epoch, step, filepath):
    print(f"\nSaving checkpoint to: {filepath}")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step
    }, filepath)


def load_checkpoint(filepath, model, optimizer):
    print(f"Loading checkpoint: {filepath}")
    ckpt = torch.load(filepath, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["step"]


def main():
    config = load_config("config.json")
    train_cfg = config["train_config"]
    data_cfg = config["data_config"]
    wg_cfg = config["waveglow_config"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------
    # 1. MEL TRANSFORM
    # -------------------------------
    mel_transform = MelSpectrogram(
        sampling_rate=data_cfg["sampling_rate"],
        n_mel_channels=wg_cfg["n_mel_channels"],
        filter_length=data_cfg["filter_length"],
        hop_length=data_cfg["hop_length"],
        win_length=data_cfg["win_length"],
        mel_fmin=data_cfg["mel_fmin"],
        mel_fmax=data_cfg["mel_fmax"]
    ).to(device)

    # -------------------------------
    # 2. DATASET
    # -------------------------------
    dataset = AudioMelDataset(
        filelist=data_cfg["training_files"],
        segment_length=data_cfg["segment_length"],
        sampling_rate=data_cfg["sampling_rate"],
        mel_transform=mel_transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    # -------------------------------
    # 3. MODEL
    # -------------------------------
    waveglow = WaveGlow(wg_cfg).to(device)
    optimizer = Adam(waveglow.parameters(), lr=train_cfg["learning_rate"])

    start_epoch, start_step = 0, 0

    if train_cfg["checkpoint_path"] != "":
        start_epoch, start_step = load_checkpoint(
            train_cfg["checkpoint_path"], waveglow, optimizer
        )

    if train_cfg["fp16_run"]:
        scaler = torch.cuda.amp.GradScaler()

    # -------------------------------
    # 4. TRAIN LOOP
    # -------------------------------
    waveglow.train()

    total_steps = start_step

    for epoch in range(start_epoch, train_cfg["epochs"]):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}")

        for mel, audio in loop:
            mel = mel.to(device)
            audio = audio.to(device)

            optimizer.zero_grad()

            # FP16
            if train_cfg["fp16_run"]:
                with torch.cuda.amp.autocast():
                    loss = waveglow.loss(mel, audio)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()

            else:
                loss = waveglow.loss(mel, audio)
                loss.backward()
                optimizer.step()

            total_steps += 1
            loop.set_postfix(loss=float(loss))

            # SAVE CKPT
            if total_steps % train_cfg["iters_per_checkpoint"] == 0:
                ckpt_path = os.path.join(
                    train_cfg["output_directory"], f"waveglow_{total_steps}.pt"
                )
                save_checkpoint(waveglow, optimizer, epoch, total_steps, ckpt_path)

        # SAVE END OF EPOCH
        ckpt_end = os.path.join(
            train_cfg["output_directory"], f"waveglow_epoch_{epoch}.pt"
        )
        save_checkpoint(waveglow, optimizer, epoch, total_steps, ckpt_end)


if __name__ == "__main__":
    main()
