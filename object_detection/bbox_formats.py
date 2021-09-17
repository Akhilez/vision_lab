"""

native:
    (xmin, ymin, xmax, ymax)
    Example: (12, 15, 43, 52)
    this is pascal voc style
    no normalization.
    no H, W required.
    no grid params required.

"""
import torch


def native_to_yolo_labels(
    bboxes: list,
    labels: list,
    num_classes: int,
    height_image: int,
    width_image: int,
    grid_rows: int,
    grid_cols: int,
) -> torch.Tensor:
    """
    Returns a tensor of shape (
        grid_rows,
        grid_cols,
        num_classes + 5
    )
    Example: (7, 7, 20 + 5)

    - Find responsible pairs.
        - cell i,j is responsible for predicting k bbox.
        - So we need a list of tuples like [(i, j, k), ...]
        - This can be based on multiple factors. One is midpoint inclusion.

    -


    - Find midpoints of all bboxes.
    - Find cell h, w, origin for all cells.
    - For each bbox:
        - Find the midpoint.
        - For each cell:
            - Continue if midpoint is not in the cell.
            - cell i,j is responsible for predicting bbox k
            - Find midpoint x, y relative to the cell origin
            - Find w and h relative to cell's w and h.
            - Record (i, j, k, x, y, w, h)
            - Break
    """


"""
You got the predictions.
Now Find IoU for responsible pairs and then set it as confidence.

At test time we multiply the conditional class probabilities and the individual box confidence predictions
"""
