# VAEcall.py
import torch
from torch import nn
from diffusers import AutoencoderKL
import logging

# Thiết lập logging để ghi lại thông tin và lỗi
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VAEWrapper(nn.Module):
    def __init__(self, ckpt_path="model/diffusion/vae-ft-ema-560000-ema-pruned.ckpt", device="cpu"):
        super(VAEWrapper, self).__init__()
        self.device = device
        self.vae = self._load_vae_model(ckpt_path)
        self.vae.to(self.device)
        self.scaling_factor = getattr(self.vae.config, 'scaling_factor', 0.18215)  # Giá trị mặc định từ Stable Diffusion
        logger.info(f"Khởi tạo VAEWrapper với scaling_factor: {self.scaling_factor}")

    def _load_vae_model(self, ckpt_path):
        try:
            logger.info(f"Đang tải VAE từ checkpoint: {ckpt_path}")

            # Khởi tạo AutoencoderKL với cấu hình từ một mô hình Stable Diffusion chuẩn
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                subfolder="vae",
                torch_dtype=torch.float32,
                use_auth_token=False
            )
            logger.info("Đã khởi tạo AutoencoderKL với cấu hình từ stabilityai/stable-diffusion-2-1")

            # Tải checkpoint .ckpt
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            logger.info("Đã tải checkpoint .ckpt thành công")

            # Ánh xạ trọng số từ checkpoint vào mô hình
            # Loại bỏ tiền tố không cần thiết (nếu có, ví dụ: "model." hoặc "vae.")
            adjusted_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                # Loại bỏ tiền tố "model." hoặc "vae." nếu có
                if key.startswith("model."):
                    new_key = key[len("model."):]
                elif key.startswith("vae."):
                    new_key = key[len("vae."):]
                adjusted_state_dict[new_key] = value

            # Tải state dict vào mô hình, bỏ qua các khóa không khớp
            vae.load_state_dict(adjusted_state_dict, strict=False)
            logger.info("Đã ánh xạ trọng số từ checkpoint vào AutoencoderKL")

            vae.eval()  # Đặt mô hình ở chế độ đánh giá
            return vae
        except Exception as e:
            logger.error(f"Không thể tải checkpoint VAE từ {ckpt_path}: {str(e)}")
            raise ValueError(f"Không thể tải checkpoint VAE từ {ckpt_path}: {str(e)}")

    def encode_to_latent(self, images):
        try:
            # Đảm bảo ảnh có định dạng đúng [B, C, H, W]
            if images.ndim != 4:
                raise ValueError(f"Đầu vào phải có shape [B, C, H, W], nhận được shape {images.shape}")

            # Chuyển ảnh về device của VAE
            images = images.to(self.device)

            with torch.no_grad():
                # Mã hóa ảnh thành latent space
                latent_dist = self.vae.encode(images).latent_dist
                latents = latent_dist.sample()  # Lấy mẫu từ phân phối latent
                latents = latents / self.scaling_factor  # Điều chỉnh scaling factor
            logger.debug(f"Đã mã hóa ảnh thành latent với shape: {latents.shape}")
            return latents
        except Exception as e:
            logger.error(f"Lỗi khi mã hóa ảnh thành latent: {str(e)}")
            raise RuntimeError(f"Lỗi khi mã hóa ảnh thành latent: {str(e)}")

    def decode_from_latent(self, latents):
        try:
            # Đảm bảo latent có định dạng đúng [B, C, H, W]
            if latents.ndim != 4:
                raise ValueError(f"Latent phải có shape [B, C, H, W], nhận được shape {latents.shape}")

            # Chuyển latent về device của VAE
            latents = latents.to(self.device)

            with torch.no_grad():
                # Điều chỉnh scaling factor trước khi giải mã
                latents_scaled = latents * self.scaling_factor
                # Giải mã từ latent space về ảnh
                images = self.vae.decode(latents_scaled).sample
            logger.debug(f"Đã giải mã latent thành ảnh với shape: {images.shape}")
            return images
        except Exception as e:
            logger.error(f"Lỗi khi giải mã từ latent: {str(e)}")
            raise RuntimeError(f"Lỗi khi giải mã từ latent: {str(e)}")

    def forward(self, images):
        latents = self.encode_to_latent(images)
        reconstructed_images = self.decode_from_latent(latents)
        return latents, reconstructed_images

def load_vae_model(ckpt_path="model/diffusion/vae-ft-ema-560000-ema-pruned.ckpt", device="cpu"):
    vae = VAEWrapper(ckpt_path=ckpt_path, device=device)
    return vae

# Tải mô hình và kiểm tra
if __name__ == "__main__":
    # Ví dụ sử dụng
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae_model = load_vae_model(ckpt_path="model/diffusion/vae-ft-ema-560000-ema-pruned.ckpt", device=device)
    logger.info("Đã tải mô hình VAE thành công!")

    # Kiểm tra với ảnh giả lập
    dummy_images = torch.randn(1, 3, 512, 512).to(device)  # Ảnh giả lập [B, C, H, W]
    latents, reconstructed = vae_model(dummy_images)
    logger.info(f"Latent shape: {latents.shape}")
    logger.info(f"Reconstructed image shape: {reconstructed.shape}")