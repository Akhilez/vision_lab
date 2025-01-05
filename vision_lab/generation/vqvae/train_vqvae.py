from os import makedirs
from os.path import dirname, join, exists

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm


PARENT_PATH = dirname(__file__)


def get_datasets():
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    )

    validation_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    )
    return training_data, validation_data


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0),
        )

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=512, embedding_dim=256)
        self.embedding.weight.data.uniform_(-1/512, 1/512)

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        b, c, h, w = x.shape
        assert c == 256, f"Expected 256 channels, got {c}"

        # Permute x to (batch_size, height, width, channels)
        x = x.permute(0, 2, 3, 1)  # (batch_size, height, width, channels)
        assert x.shape == (b, h, w, c)

        # Flatten x to (batch_size * height * width, channels)
        x = x.reshape(b * h * w, c)
        assert x.shape == (b * h * w, c)

        # Compute distances between x and embeddings
        distances = torch.cdist(x, self.embedding.weight)
        assert distances.shape == (b * h * w, 512)

        # Find the closest embedding for each pixel
        indices = torch.argmin(distances, dim=1)
        assert indices.shape == (b * h * w,)

        # Get the closest embedding for each pixel
        quantized = self.embedding(indices)
        assert quantized.shape == (b * h * w, c)

        # Reshape quantized to (batch_size, height, width, channels)
        quantized = quantized.view(b, h, w, c)
        assert quantized.shape == (b, h, w, c)

        # Permute quantized to (batch_size, channels, height, width)
        quantized = quantized.permute(0, 3, 1, 2)
        assert quantized.shape == (b, c, h, w)

        return quantized, indices


class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        """
        2 strided conv layers with stride 2 and window size 4x4
        2 residual 3x3 blocks (ReLU, 3x3 conv, ReLU, 1x1 conv) all have 256 units.
        """
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            ResidualBlock(),
            ResidualBlock(),
        )

        self.quantizer = VectorQuantizer()

        """
        2 residual 3x3 blocks (ReLU, 3x3 conv, ReLU, 1x1 conv) all have 256 units.
        2 transposed conv with stride 2, window size 4x4
        """
        self.decoder = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x_orig = x
        x = self.encoder(x)
        z, indices = self.quantizer(x)

        # Compute the loss
        loss_discrete = F.mse_loss(z, x.detach())
        loss_commitment = F.mse_loss(x, z.detach())

        # Copy the gradients from z to x
        x = x + (z - x).detach()

        x = self.decoder(x)

        loss_reconstruction = F.mse_loss(x, x_orig)

        return x, (loss_reconstruction, loss_discrete, loss_commitment), indices


def train():
    batch_size = 1024
    max_steps = 250_000
    dataloader_workers = 4
    lr = 2e-4
    # device = "cuda:0"
    device = "mps"

    training_data, validation_data = get_datasets()
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)
    val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers)

    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    step = 0
    epoch = 0
    while step < max_steps:
        loss_agg = 0.0
        entropy_agg = 0.0
        model.train()
        for x, _ in tqdm(train_loader, total=len(train_loader)):
            x = x.to(device)
            x_hat, losses, indices = model(x)
            loss_reconstruction, loss_discrete, loss_commitment = losses
            loss = loss_reconstruction + loss_discrete + 0.25 * loss_commitment
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_agg += loss.item()
            indices_one_hot = F.one_hot(indices, num_classes=512)  # (batch_size * height * width, 512)
            probabilities = indices_one_hot.float().mean(dim=0)
            entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum()
            entropy_agg += entropy.item()

            step += 1
            if step >= max_steps:
                break
        print(f"Epoch: {epoch}, global step {step}, loss: {loss_agg / len(train_loader)}, entropy: {entropy_agg / len(train_loader)}")

        # Evaluate one batch of validation data
        x, _ = next(iter(val_loader))
        x = x.to(device)
        model.eval()
        with torch.inference_mode():
            x_hat, losses, indices = model(x)
        loss_reconstruction, loss_discrete, loss_commitment = losses
        loss = loss_reconstruction + loss_discrete + 0.25 * loss_commitment
        indices_one_hot = F.one_hot(indices, num_classes=512)
        probabilities = indices_one_hot.float().mean(dim=0)
        entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum()
        print(f"Validation loss: {loss.item()}, validation entropy: {entropy.item()}")
        # Save the first 25 reconstructed images
        save_path = join(PARENT_PATH, "reconstructions", f"epoch_{epoch}")
        makedirs(save_path, exist_ok=True)
        for i in range(25):
            save_image(x_hat[i], join(save_path, f"reconstruction_{i}.jpg"))
        original_path = join(PARENT_PATH, "originals")
        if not exists(original_path):
            makedirs(original_path, exist_ok=True)
            for i in range(25):
                save_image(x[i], join(original_path, f"original_{i}.jpg"))

        epoch += 1


def test_interface():
    training_data, validation_data = get_datasets()
    print(f"Training data: {len(training_data)}")
    print(f"Validation data: {len(validation_data)}")

    x, y = training_data[0]
    print(f"Image shape: {x.shape}")
    print(f"Label: {y}")
    print(f"Image min: {x.min()}, max: {x.max()}, mean: {x.mean()}, std: {x.std()}")

    model = VQVAE()
    x_hat, losses, indices = model(x.unsqueeze(0))
    print(f"Reconstruction shape: {x_hat.shape}")
    print(f"Reconstruction min: {x_hat.min()}, max: {x_hat.max()}, mean: {x_hat.mean()}, std: {x_hat.std()}")
    print(f"Losses: {losses}")
    print(f"Indices shape: {indices.shape}")
    print(f"Indices: {indices}")

    indices_one_hot = F.one_hot(indices, num_classes=512)  # (batch_size * height * width, 512)
    probabilities = indices_one_hot.float().mean(dim=0)  # (512,)
    entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum()
    print(f"Entropy: {entropy}")


if __name__ == "__main__":
    # test_interface()
    train()