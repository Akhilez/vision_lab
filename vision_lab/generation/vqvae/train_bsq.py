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


class BSQantizer(nn.Module):
    def __init__(self, num_emb, num_bits):
        super().__init__()
        self.num_emb = num_emb
        self.num_bits = num_bits

        self.input_projection = nn.Conv2d(num_emb, num_bits, kernel_size=1, stride=1, padding=0)
        self.output_projection = nn.Conv2d(num_bits, num_emb, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        b, c, h, w = x.shape
        assert c == self.num_emb, f"Expected {self.num_emb} channels, got {c}"

        x = self.input_projection(x)

        # The core idea is to apply x/|x| and get the sign of x and multiply with 1/sqrt(num_bits) while keeping gradients.
        x = x / torch.norm(x, dim=1, keepdim=True)
        u = (x >= 0) * (1 / (self.num_bits ** 0.5))
        # Straight through gradient
        u = x + (u - x).detach()

        x = self.output_projection(u)
        assert x.shape == (b, self.num_emb, h, w)

        return x, u


def entropy(tensor, dim=-1):
    """Computes entropy using softmax-based normalization to ensure a valid probability distribution."""
    probs = torch.softmax(tensor, dim=dim)  # Always sums to 1
    entropy_val = -torch.sum(probs * probs.clamp(min=1e-10).log(), dim=dim)
    return entropy_val


class BSQVAE(nn.Module):
    def __init__(self, emb_size, num_bits):
        super().__init__()
        """
        2 strided conv layers with stride 2 and window size 4x4
        2 residual 3x3 blocks (ReLU, 3x3 conv, ReLU, 1x1 conv) all have 256 units.
        """
        self.emb_size = emb_size
        self.num_bits = num_bits

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=emb_size, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=4, stride=2, padding=1),
            ResidualBlock(emb_size),
            ResidualBlock(emb_size),
        )

        self.quantizer = BSQantizer(emb_size, num_bits)

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
        x, u = self.quantizer(x)
        x = self.decoder(x)

        loss_reconstruction = F.mse_loss(x, x_orig)
        loss_entropy = self.get_entropy_loss(u)

        return x, (loss_reconstruction, loss_entropy)

    def get_entropy_loss(self, u):
        # u: (batch_size, channels, height, width)
        b, c, h, w = u.shape
        assert c == self.num_bits, f"Expected {self.num_bits} channels, got {c}"

        # Entropy within the channels of a pixel should be low.
        # When we average all pixels (c, h, w) --> (c,), then the entropy of (c,) should be high.

        # Entropy of channels for each pixel.
        e1 = entropy(u, dim=1)  # (1, 32, 32)
        avg1 = torch.mean(e1, dim=(1, 2))  # (1,)

        # Entropy of mean of each channel
        avg2 = torch.mean(u, dim=(2, 3))  # (1, 8)
        e2 = entropy(avg2, dim=1)  # (1,)

        loss = avg1 - e2
        loss = loss.mean()
        return loss



def evaluate(val_loader, model, device, epoch):
    # Evaluate one batch of validation data
    x, _ = next(iter(val_loader))
    x = x.to(device)
    model.eval()
    with torch.inference_mode():
        x_hat, losses = model(x)
    loss_reconstruction, loss_entropy = losses
    loss = loss_reconstruction + loss_entropy
    print(f"Validation loss: {loss.item()} (Recon: {loss_reconstruction.item()}, Entropy: {loss_entropy.item()})")

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
    embedding_size = 64
    num_bits = 12

    training_data, validation_data = get_datasets()
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)
    val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers)

    model = BSQVAE(embedding_size, num_bits).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    evaluate(val_loader, model, device, epoch=-1)

    step = 0
    epoch = 0
    while step < max_steps:
        loss_recon_agg = 0.0
        loss_entropy_agg = 0.0
        model.train()
        for x, _ in tqdm(train_loader, total=len(train_loader)):
            x = x.to(device)
            x_hat, losses = model(x)
            loss_reconstruction, loss_entropy = losses
            loss = loss_reconstruction + loss_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_recon_agg += loss_reconstruction.item()
            loss_entropy_agg += loss_entropy.item()

            step += 1
            if step >= max_steps:
                break
        print(f"Epoch: {epoch}, global step {step}, loss_recon: {loss_recon_agg / len(train_loader)}, loss_entropy: {loss_entropy_agg / len(train_loader)}")

        if epoch % 5 == 0 or step >= max_steps - 1:
            evaluate(val_loader, model, device, epoch)
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
    num_bits = 9

    model = BSQVAE(embedding_size, num_bits)
    x_hat, losses = model(x.unsqueeze(0))
    print(f"Reconstruction shape: {x_hat.shape}")
    print(f"Reconstruction min: {x_hat.min()}, max: {x_hat.max()}, mean: {x_hat.mean()}, std: {x_hat.std()}")
    print(f"Losses: {losses}")


if __name__ == "__main__":
    # test_interface()
    train()
