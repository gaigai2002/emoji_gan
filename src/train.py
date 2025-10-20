import torch
from torch import nn, optim
from torchvision.utils import save_image
from .dataset import get_emoji_loader
from .model import Generator, Discriminator
import os

def train(data_dir, epochs=200, batch_size=128, z_dim=100, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_emoji_loader(data_dir, batch_size=batch_size)

    generator = Generator(z_dim=z_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optim_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    os.makedirs("output", exist_ok=True)

    for epoch in range(epochs):
        for i, imgs in enumerate(loader):
            imgs = imgs.to(device)
            batch_size = imgs.size(0)

            # ✅ 使用 label smoothing 提升训练稳定性
            real_labels = (torch.ones(batch_size, 1) * 0.9).to(device)  # soft label
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ======== Train Discriminator ========
            z = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = generator(z).detach()

            real_out = discriminator(imgs)
            fake_out = discriminator(fake_imgs)

            loss_d_real = criterion(real_out, real_labels)
            loss_d_fake = criterion(fake_out, fake_labels)
            loss_d = (loss_d_real + loss_d_fake) / 2

            optim_d.zero_grad()
            loss_d.backward()
            optim_d.step()

            # ======== Train Generator ========
            z = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = generator(z)
            fake_out = discriminator(fake_imgs)

            # ✅ 让生成器以为这些是“真实”的
            loss_g = criterion(fake_out, real_labels)

            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()

        # ✅ 打印更详细的训练信息
        print(f"Epoch [{epoch+1}/{epochs}]  Loss_D: {loss_d.item():.4f}  Loss_G: {loss_g.item():.4f}")

        # ✅ 每10轮保存一次图片（64张）
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fixed_z = torch.randn(64, z_dim).to(device)
                fake_imgs = generator(fixed_z)
                save_image(fake_imgs, f"output/fake_{epoch+1}.png", nrow=8, normalize=True)
