from os import makedirs
from os.path import dirname, join, exists
from shutil import rmtree

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import random_split
from tqdm import tqdm


PARENT_PATH = dirname(__file__)


def grayscale_to_rgb(img):
    return img.convert("RGB")


def get_datasets():
    # t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # training_data = datasets.CIFAR10(
    #     root="data",
    #     train=True,
    #     download=True,
    #     transform=t,
    # )
    #
    # validation_data = datasets.CIFAR10(
    #     root="data",
    #     train=False,
    #     download=True,
    #     transform=t,
    # )

    t = transforms.Compose([
        transforms.Lambda(grayscale_to_rgb),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data = datasets.Caltech256(
        root="data",
        transform=t,
        download=True,
    )

    # Set seed for reproducibility
    torch.manual_seed(0)
    training_data, validation_data = random_split(data, [len(data) - 1000, 1000])

    return training_data, validation_data


class ResidualBlock(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=emb_size, out_channels=emb_size * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=emb_size * 2, out_channels=emb_size, kernel_size=1, padding=0),
        )

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizer(nn.Module):
    def __init__(self, emb_size, num_embs):
        super().__init__()
        self.emb_size = emb_size
        self.num_embs = num_embs
        self.embedding = nn.Embedding(num_embeddings=num_embs, embedding_dim=emb_size)
        self.embedding.weight.data.uniform_(-1/num_embs, 1/num_embs)

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        b, c, h, w = x.shape
        assert c == self.emb_size, f"Expected {self.emb_size} channels, got {c}"

        # Permute x to (batch_size, height, width, channels)
        x = x.permute(0, 2, 3, 1)  # (batch_size, height, width, channels)
        assert x.shape == (b, h, w, c)

        # Flatten x to (batch_size * height * width, channels)
        x = x.reshape(b * h * w, c)
        assert x.shape == (b * h * w, c)

        # Compute distances between x and embeddings
        distances = torch.cdist(x, self.embedding.weight)
        assert distances.shape == (b * h * w, self.num_embs)

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
    def __init__(self, emb_size, num_emb):
        super().__init__()
        """
        2 strided conv layers with stride 2 and window size 4x4
        2 residual 3x3 blocks (ReLU, 3x3 conv, ReLU, 1x1 conv) all have 256 units.
        """
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=emb_size, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=4, stride=2, padding=1),
            ResidualBlock(emb_size),
            ResidualBlock(emb_size),
        )

        self.quantizer = VectorQuantizer(emb_size, num_embs=num_emb)

        """
        2 residual 3x3 blocks (ReLU, 3x3 conv, ReLU, 1x1 conv) all have 256 units.
        2 transposed conv with stride 2, window size 4x4
        """
        self.decoder = nn.Sequential(
            ResidualBlock(emb_size),
            ResidualBlock(emb_size),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=emb_size, out_channels=emb_size, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=emb_size, out_channels=emb_size, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=emb_size, out_channels=3, kernel_size=4, stride=2, padding=1),
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


def evaluate(val_loader, model, device, epoch, num_emb):
    # Evaluate one batch of validation data
    x, _ = next(iter(val_loader))
    x = x.to(device)
    model.eval()
    with torch.inference_mode():
        x_hat, losses, indices = model(x)
    loss_reconstruction, loss_discrete, loss_commitment = losses
    loss = loss_reconstruction + loss_discrete + 0.25 * loss_commitment
    indices_one_hot = F.one_hot(indices, num_classes=num_emb)
    probabilities = indices_one_hot.float().mean(dim=0)
    entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum()
    print(f"Validation loss: {loss.item()}, validation entropy: {entropy.item()}")
    # Save the first 25 reconstructed images
    save_path = join(PARENT_PATH, "reconstructions", f"epoch_{epoch}")
    makedirs(save_path, exist_ok=True)
    for i in range(25):
        save_image((x_hat[i] + 1) / 2, join(save_path, f"reconstruction_{i}.jpg"))
    original_path = join(PARENT_PATH, "originals")
    if epoch == -1:
        rmtree(original_path)
        makedirs(original_path, exist_ok=True)
        for i in range(25):
            save_image((x[i] + 1) / 2, join(original_path, f"original_{i}.jpg"))


def train():
    batch_size = 256
    max_steps = 250_000
    dataloader_workers = 4
    lr = 1e-3
    device = "cuda:0" if torch.cuda.is_available() else "mps"
    embedding_size = 32
    num_emb = 512

    training_data, validation_data = get_datasets()
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)
    val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers)

    model = VQVAE(embedding_size, num_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    evaluate(val_loader, model, device, epoch=-1, num_emb=num_emb)

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
            indices_one_hot = F.one_hot(indices, num_classes=num_emb)  # (batch_size * height * width, num_emb)
            probabilities = indices_one_hot.float().mean(dim=0)
            entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum()
            entropy_agg += entropy.item()

            step += 1
            if step >= max_steps:
                break
        print(f"Epoch: {epoch}, global step {step}, loss: {loss_agg / len(train_loader)}, entropy: {entropy_agg / len(train_loader)}")

        if epoch % 5 == 0 or step >= max_steps - 1:
            evaluate(val_loader, model, device, epoch, num_emb)
            checkpoints_path = join(PARENT_PATH, "checkpoints")
            makedirs(checkpoints_path, exist_ok=True)
            torch.save(model.state_dict(), join(checkpoints_path, f"model_{step}.pt"))

        epoch += 1


def test_interface():
    training_data, validation_data = get_datasets()
    print(f"Training data: {len(training_data)}")
    print(f"Validation data: {len(validation_data)}")

    x, y = training_data[0]
    print(f"Image shape: {x.shape}")
    print(f"Label: {y}")
    print(f"Image min: {x.min()}, max: {x.max()}, mean: {x.mean()}, std: {x.std()}")

    embedding_size = 64
    num_emb = 1024

    model = VQVAE(embedding_size, num_emb)
    x_hat, losses, indices = model(x.unsqueeze(0))
    print(f"Reconstruction shape: {x_hat.shape}")
    print(f"Reconstruction min: {x_hat.min()}, max: {x_hat.max()}, mean: {x_hat.mean()}, std: {x_hat.std()}")
    print(f"Losses: {losses}")
    print(f"Indices shape: {indices.shape}")
    print(f"Indices: {indices}")

    indices_one_hot = F.one_hot(indices, num_classes=num_emb)  # (batch_size * height * width, num_emb)
    probabilities = indices_one_hot.float().mean(dim=0)  # (num_emb,)
    entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum()
    print(f"Entropy: {entropy}")


if __name__ == "__main__":
    # test_interface()
    train()