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


def generate_dataset(count=10, height = 448, width = 448):
    for image_i in range(count):
        num_columns = randint(1, 3)
        for column_i in range(num_columns):
            n_rows = randint(1, 3)
            n_cols = randint(1, 2)

            cell_h = height // n_rows
            cell_w = width // n_cols

            for row_i in range(n_rows):
                for col_i in range(n_cols):
                    # If include image or not.
                    if random() > 0.5:
                        continue

                    x1, y1, x2, y2 = get_random_frame_in_cell(cell_h, cell_w, min_h=0.5, min_w=0.5, padding=0.1)
                    h, w = x2 - x1, y2 - y1


def get_random_frame_in_cell(cell_h, cell_w, min_h, min_w, padding) -> Tuple[int, int, int, int]:
    # y1 -> 0 to ((h - min_h) / 2)

    min_h = int(cell_h * min_h)
    min_w = int(cell_w * min_w)
    padding = (cell_h + cell_w) * padding // 2

    y_bound = ((cell_h - min_h) // 2)
    x_bound = ((cell_w - min_w) // 2)

    x1 = randint(padding, x_bound)
    x2 = randint(cell_w - x_bound, cell_w - padding)
    y1 = randint(padding, y_bound)
    y2 = randint(cell_h - y_bound, cell_h - padding)

    return x1, y1, x2, y2


class Coordinate(BaseModel):
    x: int
    y: int


class Painting(BaseModel):
    top_left: Coordinate
    top_right: Coordinate
    bottom_left: Coordinate
    bottom_right: Coordinate
    painting: Optional[Any]


def get_trapezoid(height, width, ratio, num_paintings):

    max_height = max_painting_height / num_paintings * 2
    max_width = min(width, max_painting_width)
    max_width = max_width / num_paintings * 2

    painting_height_1 = randint(min_painting_height, max_height)
    painting_height_2 = int(painting_height_1 * ratio)
    painting_height_2 = min(height, painting_height_2)
    longer_height = max(painting_height_1, painting_height_2)

    painting_width = randint(min_painting_width, max_width)

    start_left = randint(10, width - painting_width)
    start_top = randint(10, height - longer_height)

    mid_height = start_top + longer_height / 2

    points = [
        (start_left, mid_height - painting_height_1 // 2),
        (start_left, mid_height + painting_height_1 // 2),
        (start_left + painting_width, mid_height + painting_height_2 // 2),
        (start_left + painting_width, mid_height - painting_height_2 // 2),
    ]

    return points


def overlaps(trapezoid, trapezoids):
    return False


def get_random_painting():
    url = 'https://loremflickr.com/480/320/painting'
    img = Image.open(requests.get(url, stream=True).raw)
    img = img.convert('RGB')
    img = np.array(img)
    return img


height = 448
width = 448

min_painting_height = 50
max_painting_height = 350
min_painting_width = 50
max_painting_width = 250

image = np.zeros((3, height, width))
num_columns = np.random.randint(low=1, high=4, size=(1,))[0]

col_with = width // num_columns

for i in range(num_columns):
    start_x = i * col_with
    end_x = start_x + col_with
    section = image[:, :, start_x: end_x]
    section[:] = np.random.randint(low=0, high=255, size=(1,))[0]

    perspective_ratio = np.random.random((1,))[0] - 0.5 * 0.8
    num_paintings = np.random.randint(low=1, high=3, size=(1,))[0]

    trapezoids = []

    for j in range(num_paintings):
        for attempt in range(20):
            trapezoid = get_trapezoid(height, col_with, perspective_ratio, num_paintings)
            if not overlaps(trapezoid, trapezoids):
                trapezoids.append(trapezoid)
                break
        else:
            continue

        pts = np.array(trapezoid, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(section[0], pts=[pts], color=(255, 255, 255))
        # painting = get_random_painting()


plt.imshow(image[0])
plt.show()
