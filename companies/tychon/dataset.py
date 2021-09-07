import os
from os.path import join, dirname, basename, splitext
from typing import List
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

VALID_IMAGE_FORMATS = [".png", ".jpg", ".jpeg"]

HEIGHT = 320
WIDTH = 448


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """
    input is an image that is uint8 np.ndarray of shape (h, w, c)
    output is a torch.FloatTensor of shape (c, h, w)
    """
    image = image / 255.0
    image = np.moveaxis(image, -1, 0)
    image = torch.from_numpy(image)
    image = image.float()
    return image


def _is_image_name(file_name):
    ext = splitext(file_name)[1]
    return ext in VALID_IMAGE_FORMATS


def load_image_paths(datasets: List[str]) -> List[str]:
    """
    datasets is a list of paths of datasets. Where a dataset is a dir that contains images subdir.
    dataset:
    |--images
    |--masks
    """
    image_paths = []
    for dataset in datasets:
        for path, subdir, files in os.walk(join(dataset, "images")):
            image_paths.extend([join(path, f) for f in files if _is_image_name(f)])
    return image_paths


def preprocess_mask(image: Image) -> np.array:
    mask = np.array(image.convert("L"))
    # Does the mask contain 0-255 or 0-2?
    n_classes = len(np.unique(mask))
    if np.max(mask) > n_classes:
        mask = mask / (255.0 / (n_classes - 1))
    mask = np.round(mask).astype(np.long)
    return mask


class PaintingsDataset(Dataset):
    def __init__(self, image_paths: List[str], preprocess=None):
        super(PaintingsDataset, self).__init__()
        self.image_paths = image_paths
        self.size = len(self.image_paths)
        self.preprocess = preprocess or preprocess_image

    def __getitem__(self, i):
        image = self._get_image(self.image_paths[i])
        image = preprocess_image(image)

        mask = self._get_mask(i)
        mask = preprocess_mask(mask)

        return image, mask

    def __len__(self):
        return self.size

    @staticmethod
    def _get_image(image_path):
        image = Image.open(image_path).convert("RGB").resize((WIDTH, HEIGHT))
        image = np.array(image)
        return image

    def _get_mask(self, i):
        subdir_name = 'masks'
        extension = '.png'

        image_path = self.image_paths[i]
        image_name = splitext(basename(image_path))[0]
        file_name = image_name + extension
        parent = dirname(dirname(image_path))
        mask_path = join(parent, subdir_name, file_name)
        return Image.open(mask_path)


if __name__ == "__main__":
    dataset_path = './data'
    image_paths = load_image_paths([dataset_path])

    dataset = PaintingsDataset(image_paths)

    import matplotlib.pyplot as plt

    for image, mask in dataset[:2]:
        image = image.moveaxis(0, -1)
        print(image.shape)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[1].imshow(mask)
        plt.show()
