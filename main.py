# main.py
import torch
from VAEcall import load_vae_model
from pipeline import LatentDiffiTPipeline
from dataloader import get_imagenet_dataloader
from train import train_diffit

def main():
    # Thiết lập thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tải VAE
    vae = load_vae_model(ckpt_path="model/diffusion/vae-ft-ema-560000-ema-pruned.ckpt", device=device)

    # Tải dữ liệu
    data_dir = "dataset/archive"  # Thay bằng đường dẫn thực tế
    batch_size = 32
    dataloader, dataset = get_imagenet_dataloader(data_dir, batch_size=batch_size)
    num_classes = len(dataset.classes)

    # Khởi tạo pipeline
    pipeline = LatentDiffiTPipeline(
        vae=vae,
        img_size=256,
        patch_size=16,
        in_channels=3,
        hidden_dim=768,
        depth=2,
        heads=8,
        dim_head=64,
        num_classes=num_classes
    ).to(device)

    # Huấn luyện
    num_epochs = 10
    num_timesteps = 1000
    learning_rate = 0.00003
    train_diffit(pipeline, dataloader, num_epochs, num_timesteps, device, learning_rate)


if __name__ == "__main__":
    main()