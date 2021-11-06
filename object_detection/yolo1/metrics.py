from typing import Any
import torch
from torchmetrics import MetricCollection


class MyMetricCollection(MetricCollection):
    def update_each(self, params: dict, **kwargs: Any) -> None:
        """params is a dict where key is the metric key and the values are tuples of positional arguments.
        Keyword arguments (kwargs) will be filtered based on the signature of the individual metric.
        """
        for key, m in self.items(keep_base=True):
            if key in params:
                args = params[key]
                if type(args) is not tuple:
                    args = (args,)
                m_kwargs = m._filter_kwargs(**kwargs)
                m.update(*args, **m_kwargs)

    def compute(self):
        result = super().compute()
        self.reset()
        return result


def get_ious(preds_denorm, targets_denorm) -> torch.Tensor:
    """
    - When sum(target_[x,y,w,h]) is 0, iou is 0.
    - w_cell, h_cell = 1/S
    - w_image, h_image = 1

    - Get x1, y1, x2, y2 for predicted and target boxes.
        - x1 = midpoint_x - (width / 2)
    - find box iou

    :param preds_denorm: denormalized tensor of shape (batch, 4, B, S, S)
    :param targets_denorm: denormalized tensor of shape (batch, 4, 1, S, S)
    :return: tensor of shape (batch, B, S, S)
    """

    # Combined shape: (batch, 4, B+1, S, S)
    bboxes = torch.cat((preds_denorm, targets_denorm), dim=2)

    # Change the shape so that B+1 goes to 0 and "4" goes to last.
    # Modified shape: (B+1, batch, S, S, 4)
    bboxes = bboxes.moveaxis(2, 0).moveaxis(2, -1)

    num_boxes = preds_denorm.shape[2]

    ious = []
    for i in range(num_boxes):
        iou = _custom_ious(bboxes[i], bboxes[-1])  # shape (batch, S, S)
        ious.append(iou)
    ious = torch.stack(ious)  # shape (B, batch, S, S)
    ious = ious.moveaxis(0, 1)  # shape (batch, B, S, S)

    return ious


# PRIVATE FUNCTIONS


def _custom_ious(boxes1, boxes2) -> torch.Tensor:
    """
    Performs 1 to 1 iou
    :param boxes1: tensor of shape (*N, 4)
    :param boxes2: tensor of shape (*N, 4)
    :return: tensor of shape *N
    """
    assert boxes1.shape == boxes2.shape

    ax1 = boxes1[..., 0]
    ay1 = boxes1[..., 1]
    ax2 = boxes1[..., 2]
    ay2 = boxes1[..., 3]

    bx1 = boxes2[..., 0]
    by1 = boxes2[..., 1]
    bx2 = boxes2[..., 2]
    by2 = boxes2[..., 3]

    x1 = _max(ax1, bx1)
    y1 = _max(ay1, by1)
    x2 = _min(ax2, bx2)
    y2 = _min(ay2, by2)

    zeros = torch.zeros_like(x1)
    ones = torch.ones_like(x1)

    side_x = _max(zeros, x2 - x1 + 1)
    side_y = _max(zeros, y2 - y1 + 1)

    intersection_area = side_x * side_y

    box1_area = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    box2_area = (bx2 - bx1 + 1) * (by2 - by1 + 1)

    epsilon = 1e-7
    iou = intersection_area / (box1_area + box2_area - intersection_area + epsilon)
    iou = _min(ones, iou)  # shape (*N)
    iou[bx2 - bx1 == 0] = 0.0  # Make IoU = 0 when width = 0

    return iou


def _max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Simply finds the max off the two tensors.
    Shapes of the two tensors has to be same.
    """
    return torch.amax(torch.stack([x, y]), dim=0)


def _min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Simply finds the max off the two tensors.
    Shapes of the two tensors has to be same.
    """
    return torch.amin(torch.stack([x, y]), dim=0)
