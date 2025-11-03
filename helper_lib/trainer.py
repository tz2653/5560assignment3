import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from helper_lib.model import Generator, Discriminator

# -----------------------------
# GAN Training Function
# -----------------------------
def train_gan(epochs=1, batch_size=64, latent_dim=100):
    """
    Train a simple GAN on the MNIST dataset.
    Returns a trained Generator model.
    """

    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 3. Initialize models
    G = Generator(latent_dim=latent_dim).to(device)
    D = Discriminator().to(device)

    # 4. Loss + optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 5. Training loop
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)

            # Ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = G(z)
            validity = D(gen_imgs)
            g_loss = adversarial_loss(validity, valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(D(real_imgs), valid)
            fake_loss = adversarial_loss(D(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        print(f"[Epoch {epoch+1}/{epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")

    # Return trained Generator
    return G
