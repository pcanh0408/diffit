# data/imagenet_dataloader.py
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_imagenet_dataloader(data_dir, batch_size=32, num_workers=4):

    # Định nghĩa các biến đổi (transforms) cho ảnh
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Đảm bảo ảnh có kích thước 256x256
        transforms.ToTensor(),          # Chuyển ảnh thành tensor [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Chuẩn hóa về [-1, 1]
    ])

    # Tải dữ liệu với ImageFolder
    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader, dataset