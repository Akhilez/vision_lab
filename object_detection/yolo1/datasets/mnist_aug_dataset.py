from os.path import join

import yaml
from torch.utils.data import Dataset
import pytorch_lightning as pl


class MnistAugYoloV1Transform:
    def __init__(self):
        pass


class MnistAugDataset(Dataset):
    def __init__(self, data_path: str):
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
        # TODO: Implement.

    def __len__(self):
        return len(self.image_names)

    def _read_annotations(self, data_path: str):
        annotations_path = join(data_path, "annotations.yaml")
        with open(annotations_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)
        images = sorted(data.keys())
        return images, data


class MnistAugDataModule(pl.LightningDataModule):
    pass


if __name__ == "__main__":
    data_path = "/Users/akhil/code/vision_lab/data/mnist_detection/sample/train"
