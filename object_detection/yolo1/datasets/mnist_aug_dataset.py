from os.path import join
import albumentations as A
import cv2
import numpy as np
import yaml
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from object_detection.yolo1.datasets.transforms import transform_targets_to_yolo


class MnistAugYoloV1Transform:
    def __init__(self, h: int, w: int, augment: bool, grid_size: int):
        self.h = h
        self.w = w
        self.augment = augment
        self.grid_size = grid_size

        self.num_classes = 10
        self.albument_trasform = self._get_augmentations(h, w, augment)

    def __call__(self, image: Image, targets: dict):
        """
        :param image: PIL Image
        :param targets: dict of the format:
            height: 112
            width: 112
            image_id: 0
            annotations:
              - class: 6
                cx: 30.5
                cy: 76.5
                height: 45
                id: 2
                type: number
                width: 45
                x1: 8
                x2: 53
                y1: 54
                y2: 99
        :return: Tuple[torch.Tensor, torch.Tensor]: Image and the targets.
            Targets are of the shape: (C+5, S, S)
        """
        boxes, classes = self._transform_pre_augmentation(targets)

        transformed = self.albument_trasform(
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
    def _transform_pre_augmentation(targets: dict):
        """
        Converts a dict of the format:
            height: 112
            width: 112
            image_id: 0
            annotations:
              - class: 6
                cx: 30.5
                cy: 76.5
                height: 45
                id: 2
                type: number
                width: 45
                x1: 8
                x2: 53
                y1: 54
                y2: 99
        To the format:
            (
                [(xmin, ymin, xmax, ymax), ...],
                [3, ...]
            )
        """
        classes = []
        boxes = []
        for annotation in targets["annotations"]:
            classes.append(annotation["class"])

            box = tuple(int(annotation[key]) for key in ["x1", "y1", "x2", "y2"])
            boxes.append(box)

        return boxes, classes

    @staticmethod
    def _get_augmentations(h, w, augment):
        def normalize(x, **_):
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


class MnistAugDataset(Dataset):
    def __init__(self, data_path: str, augment: bool, grid_size: int):
        """
        So there are .png files and one annotations.yaml file.
        Read the annotations.yaml file and in get item, read the image.
        annotations.yaml is of the format:
        <image_name>:
            height: 112
            width: 112
            image_id: 0
            annotations:
              - class: 6
                cx: 30.5
                cy: 76.5
                height: 45
                id: 2
                type: number
                width: 45
                x1: 8
                x2: 53
                y1: 54
                y2: 99
        :param data_path: full path to the dataset.
        """
        self.data_path = data_path
        self.image_names, self.annotations = self._read_annotations(data_path)
        self.transform = MnistAugYoloV1Transform(
            h=112, w=112, augment=augment, grid_size=grid_size
        )

    def __getitem__(self, index: int):
        """
        Read image from data_path/image_names[i]
        Get annotation from self.annotations[i]
        Transform the annotations to tensor.
        :param index: int
        :return: Tuple[torch.Tensor, torch.Tensor]
        first tensor is the image of shape (3, 112, 112)
        second tensor is the target of shape (C+5, S, S)
        S is the grid size, C is the number of classes.
        """
        image_path = join(self.data_path, self.image_names[index])
        image = Image.open(image_path)
        annotation = self.annotations[self.image_names[index]]

        image, targets = self.transform(image, annotation)
        return image, targets

    def __len__(self):
        return len(self.image_names)

    @staticmethod
    def _read_annotations(data_path: str):
        annotations_path = join(data_path, "annotations.yaml")
        with open(annotations_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)
        images = sorted(data.keys())
        return images, data


class MnistAugDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path_train: str,
        data_path_val: str,
        data_augment: bool,
        batch_size: int,
        dataloader_num_workers: int,
        grid_size: int,
        **_
    ):
        super(MnistAugDataModule, self).__init__()

        self.data_path_train = data_path_train
        self.data_path_val = data_path_val
        self.augment = data_augment
        self.batch_size = batch_size
        self.num_workers = dataloader_num_workers
        self.grid_size = grid_size

        self.dataset_train, self.dataset_val = None, None

    def setup(self, stage=None):
        self.dataset_train = MnistAugDataset(
            data_path=self.data_path_train,
            augment=self.augment,
            grid_size=self.grid_size,
        )
        self.dataset_val = MnistAugDataset(
            data_path=self.data_path_val,
            augment=False,
            grid_size=self.grid_size,
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


if __name__ == "__main__":
    data_path_ = "/Users/akhil/code/vision_lab/data/mnist_detection/sample/train"
