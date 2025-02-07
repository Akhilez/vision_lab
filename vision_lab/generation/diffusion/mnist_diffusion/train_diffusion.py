import warnings
from os.path import dirname, abspath

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
from torchsummary import summary

from vision_lab.generation.diffusion.mnist_diffusion.unet import UNet
from vision_lab.generation.diffusion.mnist_diffusion.unet2 import Unet
from vision_lab.settings import DATA_DIR

warnings.filterwarnings("ignore")

PARENT_DIR = dirname(abspath(__file__))


def save_image(image, path):
    # image: (C, H, W)
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    image = image - image.min()
    image = image / image.max()
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(path, image)


class DiffusionProcessor:
    def __init__(self):
        self.T = 100
        self.b = np.linspace(0.001, 0.1, self.T)
        self.a = 1 - self.b
        self.a_bar = np.cumprod(self.a)

        self.a_sqrt = np.sqrt(self.a)
        self.b_sqrt = np.sqrt(self.b)
        self.a_bar_sqrt = np.sqrt(self.a_bar)
        self.one_minus_a_bar_sqrt = np.sqrt(1 - self.a_bar)

    def get_inputs_and_outputs(self, image: np.ndarray, t: int | float):
        if type(t) is float:
            assert 0 <= t <= 1
            t = int(t * self.T)
            assert 0 <= t < self.T

        noise = np.random.randn(*image.shape).astype(np.float32)
        image_t = self.a_bar_sqrt[t] * image + self.one_minus_a_bar_sqrt[t] * noise

        return image_t, noise

    def backward(self, image: torch.tensor, noise: torch.tensor, t: int | float):
        if type(t) is float:
            assert 0 <= t <= 1
            t = int(t * self.T)
            assert 0 <= t < self.T

        mean = (image - (self.b[t] / self.one_minus_a_bar_sqrt[t] * noise)) / self.a_sqrt[t]
        if t == 0:
            return mean
        else:
            variance = (1 - self.a_bar[t - 1]) / (1 - self.a_bar[t])
            variance = variance * self.b[t]
            sigma = np.sqrt(variance)

            z = torch.randn(*image.shape).to(image.device)
            image_prev = mean + (sigma * z)

            return image_prev


class MnistDiffusionDataset(Dataset):
    def __init__(self, path: str, diffuser: DiffusionProcessor, is_train: bool = True):
        super().__init__()
        self.path = path
        self.diffuser = diffuser
        self.is_train = is_train

        self.mnist_raw = MNIST(
            path,
            train=is_train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # Normalize the image.
                    transforms.Normalize((0.5,), (0.5,)),  # [0, 1] -> [-1, 1]
                    # Flatten the image.
                    # transforms.Lambda(lambda x: x.view(-1)),
                ]
            ),
        )

    def __len__(self):
        return len(self.mnist_raw)

    def __getitem__(self, idx):
        image = self.mnist_raw[idx][0]
        t = np.random.rand()

        input_image, label = self.diffuser.get_inputs_and_outputs(image, t)

        return input_image, t, label


class SqueezeBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=3, padding=1),
        )


class SqueezeDiffusionModel(nn.Sequential):
    def __init__(self, in_channels=3, w=64, depth=3, out_channels=3):
        hidden_blocks = [SqueezeBlock(w, w) for _ in range(depth)]
        super().__init__(
            SqueezeBlock(in_channels, w),
            *hidden_blocks,
            nn.Conv2d(w, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        # x: (B, C, H, W)  float
        # t: (B, 1)  long
        # output: (B, C, H, W)
        x = self.encode_t(x, t / 100)
        return super().forward(x)

    @classmethod
    def encode_t(cls, x, t):
        # x: (B, C, H, W)
        # t: (B, 1)
        # output: (B, C+1, H, W)
        t = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        return torch.cat([x, t], dim=1)


# class SimpleMnistDiffusionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         w = 512
#         self.model = nn.Sequential(
#             nn.Linear(785, w),
#             nn.ReLU(),
#             nn.Linear(w, w),
#             nn.ReLU(),
#             nn.Linear(w, w),
#             nn.Linear(w, 784),
#         )
#
#     def forward(self, x, t):
#         # x: (B, C, H, W)  float
#         # t: (B,)  long
#         # output: (B, C, H, W)
#         b, c, h, w = x.shape
#         assert t.shape == (b,)
#
#         # Flatten x to (B, C*H*W)
#         x = x.view(b, -1)
#
#         # Concatenate t to x
#         x = torch.cat([x, t.view(b, 1).float() / 100], dim=1)
#
#         y = self.model(x)  # (B, C*H*W)
#
#         # Convert y to (B, C, H, W)
#         y = y.view(b, c, h, w)
#
#         return y


def sample_one(model, diffuser: DiffusionProcessor, image: torch.tensor):
    # image: (C, H, W)
    model.eval()
    with torch.no_grad():
        for t in torch.arange(diffuser.T - 1, -1, -1, device=image.device):
            inputs = image.unsqueeze(0)  # (1, C, H, W)
            noise = model(inputs, t.view((1,)))
            image = diffuser.backward(image, noise[0], int(t))
    return image


def train():
    lr = 1e-3
    batch_size = 128
    dataloader_workers = 4
    device = "cuda:1" if torch.cuda.is_available() else "mps"

    diffuser = DiffusionProcessor()
    dataset = MnistDiffusionDataset(DATA_DIR, diffuser)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)

    # model = SimpleMnistDiffusionModel().to(device)
    model = SqueezeDiffusionModel(in_channels=2, w=64, depth=6, out_channels=1).to(device)
    # model = UNet(in_channels=2, out_channels=1, depth=4, initial_filters=32).to(device)
    # model = Unet({
    #     "im_channels": 1,
    #     "down_channels": [16, 32, 64, 128],
    #     "mid_channels": [128, 128, 64],
    #     "time_emb_dim": 128,
    #     "down_sample": [True, True, False],
    #     "num_down_layers": 2,
    #     "num_mid_layers": 2,
    #     "num_up_layers": 2,
    # }).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1000):
        loss_agg = 0.0

        model.train()
        for input_image, t, label in tqdm(dataloader):
            b, c, h, w = input_image.shape

            t = (t * diffuser.T).int().to(device)
            inputs = input_image.float().to(device)
            label = label.float().to(device)

            output = model(inputs, t)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_agg += loss.item()

        print(f"{epoch=} loss: {loss_agg / len(dataloader)}")
        if epoch % 10 == 0:
            generated = sample_one(model, diffuser, torch.randn(c, h, w).to(device))
            save_image(generated, f"{PARENT_DIR}/generated/generated_{epoch}.png")
            torch.save(model.state_dict(), f"{PARENT_DIR}/checkpoints/model_{epoch:04d}.pth")


def test_mnist_diffusion_dataset():
    dataset = MnistDiffusionDataset(DATA_DIR, DiffusionProcessor())

    input_image, t, label = dataset[0]

    x = input_image
    print(x.shape)
    print(f"{x.min()=} {x.max()=} {x.mean()=} {x.std()=}")

    x = label
    print(x.shape)
    print(f"{x.min()=} {x.max()=} {x.mean()=} {x.std()=}")

    print("t = ", t)

    # model = SqueezeDiffusionModel(in_channels=2, w=64, depth=7, out_channels=1).to(torch.float)
    # model = SimpleMnistDiffusionModel().to(torch.float)
    model = UNet(in_channels=2, out_channels=1, depth=4, initial_filters=32).to(torch.float)

    inputs = model.encode_t(input_image.unsqueeze(0), torch.tensor(t).view(1, 1)).float()

    summary(model, inputs[0].shape, device="cpu")

    x = model(inputs)
    print(x.shape)


if __name__ == "__main__":
    # test_mnist_diffusion_dataset()
    train()
