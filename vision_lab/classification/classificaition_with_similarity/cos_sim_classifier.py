import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchsummary import summary
from torchmetrics import Accuracy
from tqdm import tqdm

from vision_lab.settings import DATA_DIR


def grayscale_to_rgb(img):
    return img.convert("RGB")


def get_datasets():
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_data = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=t,
    )

    validation_data = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=t,
    )

    # t = transforms.Compose([
    #     transforms.Lambda(grayscale_to_rgb),
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # data = datasets.Caltech256(
    #     root=DATA_DIR,
    #     transform=t,
    #     download=True,
    # )
    #
    # # Set seed for reproducibility
    # torch.manual_seed(0)
    # training_data, validation_data = random_split(data, [len(data) - 1000, 1000])

    return training_data, validation_data


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super().__init__()
        self.layers = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        for i in range(depth):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout2d(0.2),
                    nn.BatchNorm2d(out_channels),
                )
            )
            in_channels = out_channels
        self.layers = nn.Sequential(*self.layers)
        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        if self.in_channels == self.out_channels:
            x = x + self.layers(x)
        else:
            x = self.layers(x)
        return self.max_pool(x)


class SimClassifier(nn.Module):
    """
    Epoch 9, Loss: 2.2362537384033203, Accuracy: 0.5314477682113647
    """
    def __init__(self, n_classes):
        super().__init__()
        depth_per_block = 2
        width = 64
        self.blocks = nn.Sequential(
            Block(3, width, depth_per_block),  # 16x16
            Block(width, width, depth_per_block),  # 8x8
            Block(width, width, depth_per_block),  # 4x4
            # Block(width, width, depth_per_block),  # 2x2
            # Block(width, width, depth_per_block),  # 1x1
        )
        self.codebook = nn.Parameter(torch.randn(n_classes, width, 4, 4), requires_grad=True)
        self._cos_sim = nn.CosineSimilarity(dim=0)

    def forward(self, x):
        x = self.blocks(x)
        y = self.codebook

        # 1. Flatten the feature dimensions (all dimensions except the first)
        # x_flat will have shape (2, 64*4*4) = (2, 1024)
        # y_flat will have shape (10, 64*4*4) = (10, 1024)
        x_flat = x.view(x.size(0), -1)
        y_flat = y.view(y.size(0), -1)

        # 2. Use broadcasting to compute pairwise cosine similarity
        # Add a dimension to x_flat -> (2, 1, 1024)
        # Add a dimension to y_flat -> (1, 10, 1024)
        # Broadcasting will make them compatible for element-wise ops -> (2, 10, 1024)
        # F.cosine_similarity computes similarity along the specified dimension (dim=2 here)
        # The result will have shape (2, 10)
        similarities = F.cosine_similarity(x_flat.unsqueeze(1), y_flat.unsqueeze(0), dim=2)

        # 3. Apply softmax to get probabilities
        # The result will have shape (2, 10)
        similarities = F.softmax(similarities, dim=1)

        return similarities


class TradClassifier(nn.Sequential):
    """
    Epoch 9, Loss: 0.8677550554275513, Accuracy: 0.7678995132446289
    """
    def __init__(self, n_classes):
        depth_per_block = 2
        width = 64
        super().__init__(
            Block(3, width, depth_per_block),  # 16x16
            Block(width, width, depth_per_block),  # 8x8
            Block(width, width, depth_per_block),  # 4x4
            # Block(width, width, depth_per_block),  # 2x2
            # Block(width, width, depth_per_block),  # 1x1
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, n_classes),
        )


def train():
    n_classes = 10
    batch_size = 128
    lr = 0.001
    device = "mps"

    training_data, validation_data = get_datasets()
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    # model = SimClassifier(n_classes).to(device)
    model = TradClassifier(n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    accuracy = Accuracy("multiclass", num_classes=n_classes).to(device)

    for epoch in range(10):
        model.train()
        for x, y in tqdm(train_loader):
            optimizer.zero_grad()
            y_hat = model(x.to(device))
            loss = criterion(y_hat, y.to(device))
            loss.backward()
            optimizer.step()
            # print(f"Epoch {epoch}, Loss: {loss.item()}")

        model.eval()
        acc = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader):
                y_hat = model(x.to(device))
                acc += accuracy(y_hat, y.to(device))

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {acc / len(val_loader)}")


def test_interface():
    n_classes = 10

    training_data, validation_data = get_datasets()
    train_loader = DataLoader(training_data, batch_size=2, shuffle=True)
    x, y = next(iter(train_loader))
    print(x.shape, y)

    # model = SimClassifier(n_classes)
    model = TradClassifier(n_classes)
    y_hat = model(x)
    print(y_hat, y_hat.shape)
    summary(model, x[0].shape, device="cpu")


if __name__ == '__main__':
    # test_interface()
    train()
