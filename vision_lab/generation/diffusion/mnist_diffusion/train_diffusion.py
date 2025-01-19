import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")


def save_image(image, path):
    image = np.transpose(image, (1, 2, 0))
    # [-1, 1] -> [0, 255]
    image = (image + 1) * 127.5
    image = image.astype(np.uint8)
    cv2.imwrite(path, image)


class DiffusionProcessor:
    def __init__(self, T: int):
        self.T = T
        self.ts = np.arange(0, T)
        extreme = 0.001
        self.b = np.linspace(extreme, 1 - extreme, T)
        self.a = 1 - self.b
        self.a_bar = np.cumprod(self.a)

    def get_inputs_and_outputs(self, image: np.ndarray, t: int):
        noise1 = np.random.randn(*image.shape).astype(np.float32)
        noise2 = np.random.randn(*image.shape).astype(np.float32)

        # image_t_prev = a_bar[t-1] * image + (1 - a_bar[t-1]) * noise1
        target_image = self.a_bar[t] / self.a[t] * image + (1 - self.a_bar[t] / self.a[t]) * noise1
        input_image = self.a[t] * target_image + self.b[t] * noise2
        # image_t = a_bar[t] * image + (a[t] - a_bar[t]) * noise1 + (1 - a[t]) * noise2

        return input_image, noise2, target_image

    def backward(self, image: np.ndarray, noise: np.ndarray, t: int):
        return 1 / self.a[t] * (image - self.b[t] * noise)


class MnistDiffusionDataset(Dataset):
    def __init__(self, path: str, T: int, is_train: bool = True):
        super().__init__()
        self.path = path
        self.T = T
        self.is_train = is_train

        self.mnist_raw = MNIST(path, train=is_train, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            # Normalize the image.
            transforms.Normalize((0.5,), (0.5,)),  # [0, 1] -> [-1, 1]
            # Flatten the image.
            # transforms.Lambda(lambda x: x.view(-1)),
        ]))

        self.diffuser = DiffusionProcessor(T)

    def __len__(self):
        return len(self.mnist_raw)

    def __getitem__(self, idx):
        image = self.mnist_raw[idx][0]
        t = int(np.random.randint(0, self.T-1, 1)[0])

        input_image, label, target_image = self.diffuser.get_inputs_and_outputs(image, t)

        return input_image, t/self.T, label, target_image


class SimpleMnistDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(785, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
        )

    def forward(self, x):
        return self.model(x)


def sample_one(model, diffuser):
    image = torch.randn(1, 784)
    model.eval()
    with torch.no_grad():
        for t in range(diffuser.T-1, 0, -1):
            inputs = torch.cat([image, torch.tensor([[t/diffuser.T]])], dim=1).float()
            noise = model(inputs)
            image = diffuser.backward(image, noise, t)
            if torch.any(torch.isnan(image)):
                print("Image is NaN")
                return None
    image = image.view(1, 28, 28)
    return image


def train():
    path = "/Users/akhildevarashetti/code/vision_lab/data"
    T = 100
    lr = 1e-3
    # dataloader_workers = 4

    dataset = MnistDiffusionDataset(path, T)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  #, num_workers=dataloader_workers)

    model = SimpleMnistDiffusionModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(10):
        loss_agg = 0.0
        for input_image, t, label, target_image in tqdm(dataloader):
            b, c, h, w = input_image.shape
            inputs = torch.cat([input_image.view(b, -1), t.view(b, 1)], dim=1).float()
            target_image = target_image.view(b, -1).float()

            output = model(inputs)
            loss = criterion(output, target_image)

            if torch.any(torch.isnan(loss)):
                print("Loss is NaN")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_agg += loss.item()

        print(f"{epoch=} loss: {loss_agg}")
        generated = sample_one(model, dataset.diffuser)
        if generated is not None:
            save_image(generated, f"/Users/akhildevarashetti/code/vision_lab/vision_lab/generation/diffusion/mnist_diffusion/generated/generated_{epoch}.png")


def test_mnist_diffusion_dataset():
    path = "/Users/akhildevarashetti/code/vision_lab/data"
    T = 10

    dataset = MnistDiffusionDataset(path, T)

    input_image, t, label, target_image = dataset[0]

    x = input_image
    print(x.shape)
    print(f"{x.min()=} {x.max()=} {x.mean()=} {x.std()=}")

    x = target_image
    print(x.shape)
    print(f"{x.min()=} {x.max()=} {x.mean()=} {x.std()=}")

    x = label
    print(x.shape)
    print(f"{x.min()=} {x.max()=} {x.mean()=} {x.std()=}")

    print(t)


if __name__ == "__main__":
    # test_mnist_diffusion_dataset()
    train()
