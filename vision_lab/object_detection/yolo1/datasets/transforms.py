from typing import Tuple, List
import torch


def transform_targets_to_yolo(
    boxes: List[Tuple[int, int, int, int]],
    classes: List[int],
    num_classes: int,
    grid_size: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """
    Converts (xmin, ymin, xmax, ymax) format to yolo format.

    - Get responsible pairs:
        - Find midpoints of all bboxes.
        - For all cells, if there's a bbox midpoint in the cell,
          that cell and bbox will go in a responsible pair list.
    - Convert coordinates from (xmin, ymin, ...) to yolo style.
    - Put everything in a tensor.

    :param w:
    :param h:
    :param grid_size:
    :param num_classes:
    :param boxes: list of tuples of (xmin, ymin, xmax, ymax)
    :param classes: list of integers
    :return: torch.Tensor of shape (C+5, S, S)
    """
    pairs: List[Tuple[int, int, int]] = _get_responsible_pairs(boxes, h, w, grid_size)
    boxes_yolo = _convert_boxes_to_yolo(boxes, pairs, h, w, grid_size)

    tensor = torch.zeros((num_classes + 5, grid_size, grid_size))
    for i, (r, c, b) in enumerate(pairs):
        tensor[classes[b], r, c] = 1.0
        tensor[num_classes, r, c] = 1.0
        for j in range(4):
            tensor[num_classes + 1 + j, r, c] = boxes_yolo[i][j]
    return tensor


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


# -----------------------------------
# PRIVATE METHODS
# -----------------------------------


def _get_responsible_pairs(
    boxes: List[Tuple[int, int, int, int]], h: int, w: int, grid_size: int
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

    cell_h = h / grid_size
    cell_w = w / grid_size

    pairs = []
    for r in range(grid_size):
        y1 = r * cell_h
        y2 = y1 + cell_h
        for c in range(grid_size):
            x1 = c * cell_w
            x2 = x1 + cell_w
            for b, (mx, my) in enumerate(midpoints):
                if x1 < mx < x2 and y1 < my < y2:
                    pairs.append((r, c, b))
    return pairs


def _convert_boxes_to_yolo(
    boxes: List[Tuple[int, int, int, int]],
    pairs: List[Tuple[int, int, int]],
    h_image: int,
    w_image: int,
    grid_size: int,
) -> List[Tuple[float, float, float, float]]:
    """
    Returns a yolo style bbox coordinates for each responsible pair.
    """
    cell_h = h_image / grid_size
    cell_w = w_image / grid_size

    yolo_boxes = []
    for r, c, b in pairs:
        xmin, ymin, xmax, ymax = boxes[b]

        w = xmax - xmin + 1
        h = ymax - ymin + 1

        tw = w / w_image
        th = h / h_image

        mx = xmin + w // 2
        my = ymin + h // 2

        origin_x = c * cell_w
        origin_y = r * cell_h

        tx = (mx - origin_x) / cell_w
        ty = (my - origin_y) / cell_h

        yolo_boxes.append((tx, ty, tw, th))

    return yolo_boxes
