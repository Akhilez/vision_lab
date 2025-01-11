import math

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchsummary import summary
from torchmetrics import Accuracy
from tqdm import tqdm

from vision_lab.settings import DATA_DIR


def grayscale_to_rgb(img):
    return img.convert("RGB")


def get_datasets():
    # t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # training_data = datasets.CIFAR10(
    #     root=DATA_DIR,
    #     train=True,
    #     download=True,
    #     transform=t,
    # )
    #
    # validation_data = datasets.CIFAR10(
    #     root=DATA_DIR,
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
        root=DATA_DIR,
        transform=t,
        download=True,
    )

    # Set seed for reproducibility
    torch.manual_seed(0)
    training_data, validation_data = random_split(data, [len(data) - 1000, 1000])

    return training_data, validation_data


def dec2bin(x: torch.tensor, bits: int, dtype=torch.uint8):
    # x.shape = (batch_size, n)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(dtype)


def bin2dec(b: torch.tensor, bits: int, dtype=torch.int64):
    # b.shape = (batch_size, bits)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1).to(dtype)


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


class BitsClassifier(nn.Module):
    def __init__(self, d):
        super().__init__()

        # depth_per_block = 2
        # width = 64
        # self.blocks = nn.Sequential(
        #     Block(3, width, 1),
        #     Block(width, width, depth_per_block),
        #     Block(width, width, depth_per_block),
        #     Block(width, width, depth_per_block),
        #     Block(width, width, depth_per_block),
        #     # Block(width, width, depth_per_block),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.Linear(width, width * 2),
        #     nn.ReLU(),
        #     nn.Linear(width * 2, d),
        # )
        # # Initialize parameters uniformly mean=0, std=0.02
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0, std=0.02)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.model.classifier = nn.Linear(1280, d)

    def forward(self, x):
        # return self.blocks(x)
        return self.model(x)


def train():
    train_bits = True
    n_classes = 257
    batch_size = 32
    epochs = 200
    lr = 1e-3
    dataloader_workers = 4
    device = "cuda:0" if torch.cuda.is_available() else "mps"
    threshold = 0.5

    if train_bits:
        d = math.ceil(math.log2(n_classes))
    else:
        d = n_classes

    train_data, val_data = get_datasets()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers)

    model = BitsClassifier(d).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if train_bits:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    accuracy = Accuracy("multilabel" if train_bits else "multiclass", num_classes=n_classes, num_labels=d).to(device)

    for epoch in range(epochs):
        loss_agg = 0.0
        accuracy_agg = 0.0
        model.train()
        for x, y in tqdm(train_loader, total=len(train_loader)):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            if train_bits:
                y = dec2bin(y, d, dtype=torch.float32)
                y_hat = torch.sigmoid(y_hat)
                y_hat_decimal = bin2dec(y_hat > threshold, d)
                y_decimal = bin2dec(y, d)
                accuracy_agg += (y_hat_decimal == y_decimal).sum().item() / y_decimal.shape[0]
            else:
                y_hat = torch.softmax(y_hat, dim=1)
                accuracy_agg += accuracy(y_hat, y).item()
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_agg += loss.item()
        print(f"Epoch {epoch}, train loss: {loss_agg / len(train_loader)}, accuracy: {accuracy_agg / len(train_loader)}", end=", ")

        # Evaluate
        loss_agg = 0.0
        accuracy_agg = 0.0
        model.eval()
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            if train_bits:
                y = dec2bin(y, d, dtype=torch.float32)
                y_hat = torch.sigmoid(y_hat)
                y_hat_decimal = bin2dec(y_hat > threshold, d)
                y_decimal = bin2dec(y, d)
                accuracy_agg += (y_hat_decimal == y_decimal).sum().item() / y_decimal.shape[0]
            else:
                y_hat = torch.softmax(y_hat, dim=1)
                accuracy_agg += accuracy(y_hat, y).item()
            loss = criterion(y_hat, y)
            loss_agg += loss.item()
        print(f"val loss: {loss_agg / len(val_loader)}, accuracy: {accuracy_agg / len(val_loader)}")



def test_interface():
    n_classes = 257
    d = math.ceil(math.log2(n_classes))

    training_data, validation_data = get_datasets()
    train_loader = DataLoader(training_data, batch_size=2, shuffle=True)
    x, y = next(iter(train_loader))
    y_bin = dec2bin(y, d)
    print(x.shape, y, y_bin)

    print(f"{n_classes=}, {d=}")
    model = BitsClassifier(d)
    y_hat = model(x)
    print(y_hat, y_hat.shape)
    summary(model, x[0].shape, device="cpu")



if __name__ == '__main__':
    test_interface()
    train()

