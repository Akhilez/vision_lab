from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from object_detection.yolo1.transforms import YoloV1Transforms


class PartialVOCDetection(VOCDetection):
    def __init__(self, size: int, **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def __len__(self):
        if self.size is None:
            return super().__len__()
        return self.size


class VocYoloDataModule(pl.LightningDataModule):
    def __init__(
        self,
        grid_size: int,
        image_height: int,
        image_width: int,
        batch_size: int,
        data_path: str,
        dataloader_num_workers: int = 0,
        data_augment=False,
        dataset_size=None,
        **_,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = dataloader_num_workers
        self.augment = data_augment
        self.dataset_size = dataset_size

        self.h = image_height
        self.w = image_width
        self.dims = (3, self.h, self.w)
        self.num_classes = 20
        self.transforms = YoloV1Transforms(
            h=self.h,
            w=self.w,
            augment=self.augment,
            num_classes=self.num_classes,
            grid_size=self.grid_size,
        )

        self.dataset_train, self.dataset_val = None, None

    def prepare_data(self):
        VOCDetection(
            root=self.data_path,
            year="2012",
            image_set="trainval",
            download=True,  # TODO: Makke it True
        )

    def setup(self, stage: Optional[str] = None):
        self.dataset_train = PartialVOCDetection(
            root=self.data_path,
            year="2012",
            image_set="train",
            download=False,
            transforms=self.transforms,
            size=self.dataset_size,
        )
        self.dataset_val = PartialVOCDetection(
            root=self.data_path,
            year="2012",
            image_set="val",
            download=False,
            transforms=self.transforms,
            size=self.dataset_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
