"""
The vanilla VOC dataset yields
"""

from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from typing import List, Tuple
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from object_detection.yolo1.constants import VOC_CLASSES


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

        targets = self.transform_targets_to_yolo(boxes, classes)
        return image, targets

    def transform_targets_to_yolo(self, boxes, classes) -> torch.Tensor:
        """
        Converts (xmin, ymin, xmax, ymax) format to yolo format.

        - Get responsible pairs:
            - Find midpoints of all bboxes.
            - For all cells, if there's a bbox midpoint in the cell,
              that cell and bbox will go in a responsible pair list.
        - Convert coordinates from (xmin, ymin, ...) to yolo style.
        - Put everything in a tensor.

        :param boxes: list of tuples of (xmin, ymin, xmax, ymax)
        :param classes: list of integers
        :return: torch.Tensor of shape (C+5, S, S)
        """
        pairs: List[Tuple[int, int, int]] = self._get_responsible_pairs(boxes)
        boxes_yolo = self._convert_boxes_to_yolo(boxes, pairs)

        tensor = torch.zeros((self.num_classes + 5, self.grid_size, self.grid_size))
        for i, (r, c, b) in enumerate(pairs):
            tensor[classes[b], r, c] = 1.0
            tensor[self.num_classes, r, c] = 1.0
            for j in range(4):
                tensor[self.num_classes + 1 + j, r, c] = boxes_yolo[i][j]
        return tensor

    @staticmethod
    def transform_targets_from_yolo(
        preds: torch.Tensor, image_height: int, image_width: int, num_classes: int
    ) -> torch.Tensor:
        """
        Converts from cell-relative bboxes (x, y, w, h) to (xmin, ymin, xmax, ymax).

        Find origins
        Split xs, ys, ws, hs.
        Find midpoints: origin_x + cell_w * x
        Find box width & height: w * image_width
        Find xmin, ymin, xmax, ymax: mx - bw/2


        preds: shape (batch, (C + 5B), S, S)
        returns: shape (batch, 4, B, S, S)
        """
        batch_size, channels, grid_rows, grid_cols = preds.shape
        num_boxes = (channels - num_classes) // 5
        cell_h = image_height / grid_rows
        cell_w = image_width / grid_cols

        origin_x = (
            torch.arange(start=0, end=image_width, step=cell_w)
            .view((1, grid_cols))
            .expand((grid_rows, grid_cols))
        )
        origin_y = (
            torch.arange(start=0, end=image_height, step=cell_h)
            .view((grid_rows, 1))
            .expand((grid_rows, grid_cols))
        )

        idx = [], [], [], []
        idx_x, idx_y, idx_w, idx_h = idx

        for i in range(num_boxes):
            for j in range(4):
                idx[j].append(num_classes + (i * 5) + j + 1)

        xs = preds[:, idx_x, :, :]
        ys = preds[:, idx_y, :, :]
        ws = preds[:, idx_w, :, :]
        hs = preds[:, idx_h, :, :]

        mx = origin_x + cell_w * xs
        my = origin_y + cell_h * ys
        bw = ws * image_width
        bh = hs * image_height
        bw_half = bw // 2
        bh_half = bh // 2

        xmin = mx - bw_half
        ymin = my - bh_half
        xmax = mx + bw_half
        ymax = my + bh_half

        bboxes = torch.stack([xmin, ymin, xmax, ymax]).moveaxis(0, 1)
        return bboxes

    def _convert_boxes_to_yolo(
        self,
        boxes: List[Tuple[int, int, int, int]],
        pairs: List[Tuple[int, int, int]],
    ) -> List[Tuple[float, float, float, float]]:
        """
        Returns a yolo style bbox coordinates for each responsible pair.
        """
        cell_h = self.h / self.grid_size
        cell_w = self.w / self.grid_size

        yolo_boxes = []
        for r, c, b in pairs:
            xmin, ymin, xmax, ymax = boxes[b]

            w = xmax - xmin + 1
            h = ymax - ymin + 1

            tw = w / self.w
            th = h / self.h

            mx = xmin + w // 2
            my = ymin + h // 2

            origin_x = c * cell_w
            origin_y = r * cell_h

            tx = (mx - origin_x) / cell_w
            ty = (my - origin_y) / cell_h

            yolo_boxes.append((tx, ty, tw, th))

        return yolo_boxes

    def _get_responsible_pairs(
        self,
        boxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int]]:
        """
        - Find midpoints of all bboxes.
        - For all cells, if there's a bbox midpoint in the cell,
          that cell and bbox will go in a responsible pair list.
        """
        midpoints = []
        for (xmin, ymin, xmax, ymax) in boxes:
            x = (xmin + xmax + 1) / 2
            y = (ymin + ymax + 1) / 2
            midpoints.append((x, y))

        cell_h = self.h / self.grid_size
        cell_w = self.w / self.grid_size

        pairs = []
        for r in range(self.grid_size):
            y1 = r * cell_h
            y2 = y1 + cell_h
            for c in range(self.grid_size):
                x1 = c * cell_w
                x2 = x1 + cell_w
                for b, (mx, my) in enumerate(midpoints):
                    if x1 < mx < x2 and y1 < my < y2:
                        pairs.append((r, c, b))
        return pairs

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
