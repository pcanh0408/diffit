# train/train_diffit.py
import torch
import torch.optim as optim
import torchvision.utils as vutils
import torch.optim as optim

def train_diffit(pipeline, dataloader, num_epochs, num_timesteps, device, learning_rate):
    optimizer = optim.Adam(pipeline.parameters(), lr= learning_rate)
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        pipeline.train()
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.shape[0]

            # Tạo timestep ngẫu nhiên
            timesteps = torch.randint(0, num_timesteps, (batch_size,), device=device).float()

            # Mã hóa ảnh thành latent space
            with torch.no_grad():
                latents = pipeline.encoder.encode_to_latent(images)  # [B, 4, H, W]

            # Thêm nhiễu vào latent space (DDPM đơn giản)
            noise = torch.randn_like(latents)
            t = timesteps / num_timesteps
            noisy_latents = (1 - t.view(-1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1) * noise

            # Dự đoán nhiễu trong latent space
            optimizer.zero_grad()
            predicted_noise = pipeline(noisy_latents, timesteps, labels)  # [B, 4, H, W]
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        torch.save(pipeline.state_dict(), f"latent_diffit_epoch_{epoch+1}.pth")

        pipeline.eval()
        with torch.no_grad():
            num_samples = 8
            generated_images = pipeline.sample(num_samples=num_samples, timesteps=num_timesteps, device=device)
            generated_images = (generated_images + 1) / 2
            vutils.save_image(generated_images, f"generated_images/epoch_{epoch+1}.png", nrow=4)