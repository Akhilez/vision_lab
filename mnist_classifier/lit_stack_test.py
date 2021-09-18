from typing import Optional, List, Union
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST


class MnistDataset(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_path: str,
        val_split: float,
        dataloader_num_workers: int = 0,
        **_,
    ):
        super().__init__()

        self.h = 28
        self.w = 28
        self.dims = (1, self.h, self.w)

        self.data_path = data_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = dataloader_num_workers

        self.transform_train_val, self.transform_test = self._get_transforms()

        self.mnist_train, self.mnist_val, self.mnist_test = None, None, None

    def prepare_data(self):
        # download
        MNIST(self.data_path, train=True, download=True)
        MNIST(self.data_path, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """
        :param stage: One of {fit, validate, test}. None = all 3
        :return:
        """

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(
                self.data_path, train=True, transform=self.transform_train_val
            )
            full_size = len(mnist_full)
            val_size = int(full_size * self.val_split)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full,
                [full_size - val_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_path, train=False, transform=self.transform_test
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _get_transforms(self):
        resize_transform = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=self.h, min_width=self.w, border_mode=cv2.BORDER_CONSTANT
                ),
                A.RandomCrop(self.h, self.w),
            ]
        )
        augmentations = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ]
        )

        def _normalizer(image, **_):
            return image / 255.0

        compatible = A.Compose(
            [
                ToTensorV2(always_apply=True),
                A.Lambda(image=_normalizer),
            ]
        )

        transforms_train_val_ = A.Compose([resize_transform, augmentations, compatible])
        transforms_test_ = A.Compose([resize_transform, compatible])

        def transforms_train_val(image):
            return transforms_train_val_(image=np.array(image))["image"]

        def transforms_test(image):
            return transforms_test_(image=np.array(image))["image"]

        return transforms_train_val, transforms_test


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class SimpleCNN(nn.Module):
    def __init__(
        self,
        architecture: List[Union[tuple, str, list]],
        in_channels: int,
    ):
        super(SimpleCNN, self).__init__()
        layers = []
        for module in architecture:
            if type(module) is tuple:
                layers.append(self._get_cnn_block(module, in_channels))
                in_channels = module[1]
            elif module == "M":
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=(2, 2),
                        stride=(2, 2),
                    )
                )
            elif type(module) is list:
                for i in range(module[-1]):
                    for j in range(len(module) - 1):
                        layers.append(self._get_cnn_block(module[j], in_channels))
                        in_channels = module[j][1]
        self.model = nn.Sequential(*layers)

    @staticmethod
    def _get_cnn_block(module: tuple, in_channels):
        kernel_size, filters, stride, padding = module
        return CNNBlock(
            in_channels,
            filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        return self.model(x)


class MnistModel(pl.LightningModule):
    def __init__(self, **hp):
        super().__init__()
        architecture = [
            # (kernel_size, filters, stride, padding)
            (3, 10, 1, 1),
            "M",
            (3, 20, 1, 1),
            "M",
            (3, 40, 1, 1),
            # "M",
            (3, 80, 1, 1),
            (3, 10, 1, 0),
        ]

        self.model = nn.Sequential(
            SimpleCNN(architecture, in_channels=1),
            nn.AvgPool2d(kernel_size=5),
            nn.Flatten(),
        )

        self.hp = hp
        self.save_hyperparameters(hp)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log("train/loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._evaluate(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._evaluate(batch, batch_idx, "test")

    def _evaluate(self, batch, batch_idx, prefix: str):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log(f"{prefix}/loss", loss, on_epoch=True)
        return loss


def main():

    hp = {
        "epochs": 100,
        "lr_initial": 0.0001,
        "lr_decay_every": 30,
        "lr_decay_by": 0.3,
    }

    config = {
        "data_path": "../data",
        "val_split": 0.3,
        "batch_size": 64,
        "output_path": "./output",
        "model_save_frequency": 5,
        "dataloader_num_workers": 0,
    }

    dataset = MnistDataset(**config)
    model = MnistModel(**hp, **config)
    trainer = pl.Trainer(
        gpus=0,
        max_epochs=hp["epochs"],
        default_root_dir=config["output_path"],
    )

    trainer.fit(model, datamodule=dataset)


def test_modules():
    dataset = MnistDataset(batch_size=2)
    dataset.prepare_data()
    dataset.setup()
    loader = dataset.train_dataloader()
    for image, classes in loader:
        print(image.shape, classes)
        break

    model = MnistModel()
    y = model(image)
    print(y.shape)


if __name__ == "__main__":
    main()
