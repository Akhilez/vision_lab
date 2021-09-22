"""
This is image classification with few good practices.

Uses the following frameworks:

- PyTorch Lightning
- TorchMetrics
-
"""


from typing import Optional, List, Union
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchmetrics import AverageMeter
from torchvision.datasets import MNIST
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F


class MnistDataset(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_path: str,
        val_split: float,
        dataloader_num_workers: int = 0,
        manual_seed=None,
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
        self.manual_seed = manual_seed

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
            generator = None
            if self.manual_seed:
                generator = torch.Generator().manual_seed(self.manual_seed)

            self.mnist_train, self.mnist_val = random_split(
                mnist_full,
                [full_size - val_size, val_size],
                generator=generator,
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
        resize_transform: list = [
            A.PadIfNeeded(
                min_height=self.h, min_width=self.w, border_mode=cv2.BORDER_CONSTANT
            ),
            A.RandomCrop(self.h, self.w),
        ]

        augmentations: list = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ]

        def _normalizer(image, **_):
            return image / 255.0

        compatible: list = [
            ToTensorV2(always_apply=True),
            A.Lambda(image=_normalizer),
        ]

        transforms_train_val_ = A.Compose(resize_transform + compatible)
        transforms_test_ = A.Compose(resize_transform + compatible)

        def transforms_train_val(image):
            return transforms_train_val_(image=np.array(image))["image"]

        def transforms_test(image):
            return transforms_test_(image=np.array(image))["image"]

        return transforms_train_val, transforms_test


# %%


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


# %%


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
            # (3, 80, 1, 1),
            (3, 10, 1, 0),
        ]

        self.model = nn.Sequential(
            SimpleCNN(architecture, in_channels=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.hp = hp
        self.save_hyperparameters(hp)

        # Metrics
        self.accuracy_train = pl.metrics.Accuracy()
        self.accuracy_val = pl.metrics.Accuracy()
        self.loss_train = AverageMeter()
        self.loss_val = AverageMeter()

    def forward(self, x):
        return self.model(x)

    def _criterion(self, preds, targets):
        return F.nll_loss(F.softmax(preds, dim=1), targets)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hp["lr_initial"])

    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = self._criterion(preds, targets)
        return {
            "loss": loss,
            "preds": preds.detach(),
            "inputs": images,
            "targets": targets,
        }

    def training_step_end(self, outs: dict):
        self.loss_train(outs["loss"])
        self.accuracy_train(outs["preds"], outs["targets"])
        self.log("train/step/accuracy", self.accuracy_train, prog_bar=True)

    def training_epoch_end(self, outs: dict):
        self.log("train/epoch/accuracy", self.accuracy_train.compute())
        self.log("train/epoch/loss", self.loss_train.compute())

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = loss = self._criterion(preds, targets)
        return {
            "loss": loss,
            "preds": preds.detach(),
            "inputs": images,
            "targets": targets,
        }

    def validation_step_end(self, outs: dict):
        self.accuracy_val(outs["preds"], outs["targets"])
        self.loss_val(outs["loss"])

    def on_validation_epoch_end(self) -> None:
        self.log("val/epoch/accuracy", self.accuracy_val.compute())
        self.log("val/epoch/loss", self.loss_val.compute())
        self.loss_val.reset()


# %%


def debug1():
    config = {
        "data_path": "../data",
        "val_split": 0.3,
        "batch_size": 2,
        "output_path": "./output",
        "model_save_frequency": 5,
        "dataloader_num_workers": 0,
    }
    dataset = MnistDataset(**config)

    dataset.prepare_data()
    dataset.setup()
    loader = dataset.train_dataloader()
    for image, classes in loader:
        print(image.shape, classes.shape)
        break

    hp = {
        "epochs": 10,
        "lr_initial": 0.0001,
        "lr_decay_every": 30,
        "lr_decay_by": 0.3,
    }
    model = MnistModel(**hp, **config)
    y = model(image)
    print(y.shape)


# debug1()


# %%


def train():
    hp = {
        "epochs": 10,
        "lr_initial": 0.001,
        "lr_decay_every": 30,
        "lr_decay_by": 0.3,
    }

    config = {
        "data_path": "../data",
        "val_split": 0.05,
        "batch_size": 64,
        "manual_seed": 2,
        "output_path": "./output",
        "model_save_frequency": 5,
        "dataloader_num_workers": 0,
    }

    dataset = MnistDataset(**config)
    model = MnistModel(**hp, **config)
    wandb_logger = WandbLogger(project="classification_test", log_model=True)
    trainer = pl.Trainer(
        gpus=0,
        max_epochs=hp["epochs"],
        default_root_dir=config["output_path"],
        logger=wandb_logger,
    )
    wandb_logger.watch(model)

    trainer.fit(model, datamodule=dataset)


if __name__ == "__main__":
    train()
