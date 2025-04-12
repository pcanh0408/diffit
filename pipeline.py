import torch
import torch.nn as nn
from VAEcall import load_vae_model
from einops import rearrange
from latent_diffit import Encoder, LatentDiffiTTransformer, Unpatchify, Decoder

class LatentDiffiTPipeline(nn.Module):
    def __init__(
        self,
        vae,
        img_size=256,
        patch_size=16,
        in_channels=3,
        hidden_dim=1152,
        depth=30,
        heads=16,
        dim_head=64,
        mlp_dim=None,
        dropout=0.0,
        time_embed_dim=None,
        label_embed_dim=None,
        num_classes=1000
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        if mlp_dim is None:
            mlp_dim = hidden_dim * 4

        # Khởi tạo các thành phần
        self.encoder = Encoder(
            vae=vae,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_dim=hidden_dim
        )
        self.transformer = LatentDiffiTTransformer(
            dim=hidden_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            time_embed_dim=time_embed_dim,
            label_embed_dim=label_embed_dim,
            num_classes=num_classes
        )
        self.unpatchify = Unpatchify(patch_size=patch_size, hidden_dim=self.hidden_dim)  # Thêm hidden_dim
        self.decoder = Decoder(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim // 2,
            out_channels=4  # Khớp với số kênh của latent space
        )
        self.vae = vae

    def forward(self, noisy_latents, timesteps, labels=None):
        noisy_latents[:, :3, :, :] *= (1 + noisy_latents[:, :3, :, :]) # Apply classifier-free guidance to the first three input channels
        # Đầu vào là noisy_latents [B, 4, H, W], không cần mã hóa lại
        # Chuyển noisy_latents thành dạng phù hợp để đưa vào encoder
        embedded = self.encoder.patch_embedding(noisy_latents)  # [B, hidden_dim, H/patch_size, W/patch_size]
        embedded = rearrange(embedded, 'b c h w -> b (h w) c') + self.encoder.position_embedding
        transformer_output = self.transformer(embedded, timesteps, labels)
        unpatched = self.unpatchify(transformer_output)  # [B, hidden_dim, H, W]
        predicted_noise = self.decoder(unpatched)  # [B, 4, H, W]
        return predicted_noise

    def sample(self, num_samples, timesteps, device, labels=None):
        latent_size = self.img_size // 8
        latents = torch.randn(num_samples, 4, latent_size, latent_size).to(device)
        timesteps_tensor = torch.arange(timesteps - 1, -1, -1, device=device).float()
        if labels is None:
            labels = torch.randint(0, self.num_classes, (num_samples,), device=device)

        for t in timesteps_tensor:
            t_batch = t.repeat(num_samples).float()
            predicted_noise = self.forward(latents, t_batch, labels)
            # Cập nhật latents theo quy trình diffusion (DDPM đơn giản)
            alpha = 1 - t / timesteps  # Đây là một ví dụ đơn giản, cần điều chỉnh theo lịch trình DDPM thực tế
            latents = (latents - (1 - alpha) * predicted_noise) / alpha

        with torch.no_grad():
            images = self.vae.decode_from_latent(latents.permute(0, 2, 3, 1))
        return images