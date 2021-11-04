"""
The vanilla VOC dataset yields
"""

from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from typing import Tuple
import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from object_detection.yolo1.constants import VOC_CLASSES
from object_detection.yolo1.datasets.transforms import transform_targets_to_yolo


class VOCYoloV1Transforms:
    def __init__(self, h: int, w: int, augment: bool, num_classes: int, grid_size: int):
        self.h = h
        self.w = w
        self.augment = augment
        self.num_classes = num_classes
        self.grid_size = grid_size

        self.albument_transforms = self._get_augmentations(self.h, self.w, self.augment)

    def __call__(self, image, targets: dict):
        """
        The transform function takes in pil image and a dict of target bboxes.
        It applies augmentations and returns an image and target tensor of shape (C+5, S, S)
        The transform will return image tensor and target tensor.

        The target is of the shape excluding unrelated info:
        ```
        annotation:
          object:
            - name: bicycle
              bndbox:
                xmax: 471
                xmin: 54
                ymax: 336
                ymin: 39
        ```
        The output target will be a tensor of shape: (C+5, S, S)
        :return: Callable function
        """
        boxes, classes = self._transform_pre_augmentation(targets)

        transformed = self.albument_transforms(
            image=np.array(image),
            bboxes=boxes,
            class_labels=classes,
        )

        image = transformed["image"]
        boxes = transformed["bboxes"]
        classes = transformed["class_labels"]

        targets = transform_targets_to_yolo(
            boxes,
            classes,
            num_classes=self.num_classes,
            grid_size=self.grid_size,
            h=self.h,
            w=self.w,
        )
        return image, targets

    @staticmethod
    def _get_augmentations(h, w, augment: bool):
        def normalize(x, **kwargs):
            return x / 255.0

        resizing: list = [
            # A.LongestMaxSize(max_size=WIDTH, always_apply=True),
            A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT),
            A.RandomCrop(h, w),
            # A.Resize(height=HEIGHT, width=WIDTH, always_apply=True),
        ]
        compatibility: list = [
            ToTensorV2(always_apply=True),
            A.Lambda(image=normalize),
        ]

        augmentations: list = []
        if augment:
            augmentations = [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ]

        return A.Compose(
            resizing + augmentations + compatibility,
            bbox_params=A.BboxParams(
                format="pascal_voc", min_visibility=0.05, label_fields=["class_labels"]
            ),
        )

    @staticmethod
    def _transform_pre_augmentation(targets: dict) -> Tuple[list, list]:
        """
        This converts the targets compatible with albumentations
        The target is of the shape excluding unrelated info:
        ```
        annotation:
          object:
            - name: bicycle
              bndbox:
                xmax: 471
                xmin: 54
                ymax: 336
                ymin: 39
        ```
        Output will be of the form:
        (
            [(xmin, ymin, xmax, ymax), ...],
            [3, ...]
        )
        """
        classes = []
        boxes = []
        for object in targets["annotation"]["object"]:
            class_index = VOC_CLASSES.index(object["name"])
            classes.append(class_index)

            box = object["bndbox"]
            box = tuple(int(box[key]) for key in ["xmin", "ymin", "xmax", "ymax"])
            boxes.append(box)

        return boxes, classes


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
        self.transforms = VOCYoloV1Transforms(
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
