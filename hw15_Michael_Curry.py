import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import os

torch.set_num_threads(4)

# Parameters
latent_dims = 100
num_epochs = 20
batch_size = 32
learning_rate = 2e-4
sample_interval = 5
interpolation_steps = 10

os.makedirs("samples", exist_ok=True)

img_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 64, 4, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, 4, 3),
            nn.Tanh()
        )

    def forward(self, input):
        return self.gen(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, 16, 4, 4, 3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.disc(input)

def interpolate_latent(generator, z_dim, device):
    z1 = torch.randn(1, z_dim, 1, 1, device=device)
    z2 = torch.randn(1, z_dim, 1, 1, device=device)
    imgs = []
    for alpha in np.linspace(0, 1, interpolation_steps):
        z = z1 * (1 - alpha) + z2 * alpha
        with torch.no_grad():
            img = generator(z).detach().cpu()
        imgs.append(img)
    grid = make_grid(torch.cat(imgs, dim=0), nrow=interpolation_steps, normalize=True)
    save_image(grid, "samples/interpolation.png")

if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    dataset = MNIST(root='.', train=True, transform=img_transform, download=True)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    generator = Generator(z_dim=latent_dims).to(device)
    discriminator = Discriminator().to(device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    label_real = torch.ones(batch_size, 1, device=device)
    label_fake = torch.zeros(batch_size, 1, device=device)

    gen_losses = []
    disc_losses = []

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        gen_loss_epoch = 0.0
        disc_loss_epoch = 0.0
        for image_batch, _ in train_dataloader:
            image_batch = image_batch.to(device)

            # Training The Discriminator
            latent = torch.randn(batch_size, latent_dims, 1, 1, device=device)
            fake_images = generator(latent)

            real_pred = discriminator(image_batch)
            fake_pred = discriminator(fake_images.detach())

            disc_loss = F.binary_cross_entropy(real_pred, label_real[:len(real_pred)]) + \
                        F.binary_cross_entropy(fake_pred, label_fake[:len(fake_pred)])

            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

            # Training The Generator
            fake_pred = discriminator(fake_images)
            gen_loss = F.binary_cross_entropy(fake_pred, label_real[:len(fake_pred)])

            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            gen_loss_epoch += gen_loss.item()
            disc_loss_epoch += disc_loss.item()

        gen_losses.append(gen_loss_epoch / len(train_dataloader))
        disc_losses.append(disc_loss_epoch / len(train_dataloader))

        if (epoch + 1) % sample_interval == 0 or epoch == 0:
            with torch.no_grad():
                sample_z = torch.randn(16, latent_dims, 1, 1, device=device)
                sample_imgs = generator(sample_z)
                save_image(make_grid(sample_imgs, nrow=4, normalize=True), f"samples/sample_epoch_{epoch+1}.png")

    # Plot & show loss
    plt.figure()
    plt.plot(gen_losses, label="Generator Loss")
    plt.plot(disc_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Training Loss")
    plt.tight_layout()
    plt.savefig("samples/loss_plot.png")
    plt.show()

    interpolate_latent(generator, latent_dims, device)
