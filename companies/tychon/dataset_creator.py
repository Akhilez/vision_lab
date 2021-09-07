from random import randint, random
from typing import Optional, Any, Tuple

import cv2
import numpy as np
import requests
from matplotlib import pyplot as plt
from pydantic import BaseModel
from PIL import Image


# 160 224
# 320 448
# 480 672
# 640 896

# https://loremflickr.com/1280/720/painting
from companies.tychon.rotate_3d import rotate_y

"""
- Divide the image into 1-3 columns
- For each column:
    - select number of frames / grid size.
    - for each frame:
        - Find (h, w) in from (cell_h * 0.5) to (cell_h * 0.9).
        - place a random painting.
    - Select if angled or flat.
    - if angled, randomly select angle.
"""


def generate_dataset(count=10, height=448, width=448):
    for image_i in range(count):
        num_columns = randint(1, 3)

        image = np.zeros((4, height, width))
        col_width = width // num_columns

        for column_i in range(num_columns):
            start_x = column_i * col_width
            end_x = start_x + col_width
            section = image[:, :, start_x: end_x]
            section[:] = randint(0, 255)

            n_rows = randint(1, 4)
            n_cols = randint(1, 2)

            cell_h = height // n_rows
            cell_w = col_width // n_cols

            for row_i in range(n_rows):
                for col_i in range(n_cols):
                    dont_include = random() > 0.8
                    if dont_include:
                        continue

                    x1, y1, x2, y2 = get_random_frame_in_cell(
                        cell_h,
                        cell_w,
                        min_h=0.5,
                        min_w=0.5,
                        padding=0.1
                    )
                    h, w = y2 - y1, x2 - x1

                    x1 = (col_i * cell_w) + x1
                    x2 = (col_i * cell_w) + x2
                    y1 = (row_i * cell_h) + y1
                    y2 = (row_i * cell_h) + y2

                    painting = get_random_painting(h, w)
                    section[: 3, y1: y2, x1: x2] = painting
                    section[3, y1: y2, x1: x2] = np.ones((h, w))

            is_angled = random() > 1 / 3
            if is_angled:
                degree = randint(10, 60)
                section_ = np.moveaxis(section, 0, -1)
                rotated = rotate_y(section_, degree)
                section_[:] = rotated[:]

        plt.imshow(image[0])
        plt.show()


def get_random_painting(height, width):
    return np.ones((3, height, width)) * randint(0, 255)


def get_random_frame_in_cell(cell_h, cell_w, min_h, min_w, padding) -> Tuple[int, int, int, int]:
    # y1 -> 0 to ((h - min_h) / 2)

    min_h = int(cell_h * min_h)
    min_w = int(cell_w * min_w)
    padding = int(min(cell_h, cell_w) * padding)

    y_bound = ((cell_h - min_h) // 2)
    x_bound = ((cell_w - min_w) // 2)

    x1 = randint(padding, x_bound)
    x2 = randint(cell_w - x_bound, cell_w - padding)
    y1 = randint(padding, y_bound)
    y2 = randint(cell_h - y_bound, cell_h - padding)

    return x1, y1, x2, y2


generate_dataset(5, height=300, width=500)
