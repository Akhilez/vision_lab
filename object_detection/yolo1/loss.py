from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F
from object_detection.yolo1.datasets.transforms import transform_targets_from_yolo


class YoloV1Loss(nn.Module):
    """
    Losses:

    - Object exists: lambda_coord * sum((x - xhat)^2 + (y - yhat)^2)
    - Object exists: lambda_coord * sum((sqrt(w) - sqrt(w_hat))^2 + (sqrt(h) - sqrt(h_hat))^2)
    - Object exists: 1 * sum((confidence - confidence_hat)^2)
    - No-object exists: lambda_no_object * sum((confidence - confidence_hat)^2)
    - Object exists: sum((probability(c) - probability(c_hat))^2)

    confidence = IoU
    lambda_coord = 5
    lambda_no_object = 0.5
    """

    def __init__(
        self,
        num_boxes: int,
        num_classes: int,
        image_height: int,
        image_width: int,
        lambda_coord: float,
        lambda_object_exists: float,
        lambda_no_object: float,
        lambda_class: float,
    ):
        """
        Find the responsible cell-bbox pairs.

        :param num_boxes: (B)
        :param num_classes: (C)
        """
        super().__init__()

        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.image_height = image_height
        self.image_width = image_width

        self.lambda_coord = lambda_coord
        self.lambda_object_exists = lambda_object_exists
        self.lambda_no_object = lambda_no_object
        self.lambda_class = lambda_class

        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        ious: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        - Responsible box is the one that has the highest IoU.

        IoUs is a 0-1 tensor of shape (batch, B, S, S)
        Responsibility is an index tensor of shape (batch, S, S)
        object_exists is a 0,1 tensor of shape (batch, 1, S, S)

        :param preds: tensor of shape (batch, (C + B * 5), S, S)
        :param targets: tensor of shape (batch, C+5, S, S)
        :param ious: torch.Tensor of shape (batch, B, S, S)
        :return: a dict of all losses.
        """

        # shape: (batch, S, S)
        object_exists = targets[:, self.num_classes]

        # shape (batch, B, S, S)
        responsibility = (
            F.one_hot(ious.argmax(dim=1), num_classes=self.num_boxes)
            .float()
            .moveaxis(-1, 1)
        )
        for i in range(self.num_boxes):
            responsibility[:, i] *= object_exists

        coords_loss = self._get_coords_loss(preds, targets, responsibility)
        confidence_loss = self._get_confidence_loss(preds, ious, responsibility)
        negative_confidence_loss = self._get_confidence_loss(
            preds, torch.zeros_like(ious), 1 - responsibility
        )
        class_loss = self._get_class_loss(preds, targets, object_exists)

        final_loss = (
            coords_loss * self.lambda_coord
            + confidence_loss * self.lambda_object_exists
            + negative_confidence_loss * self.lambda_no_object
            + class_loss * self.lambda_class
        )

        return {
            "loss": final_loss,
            "loss_coords": coords_loss.detach(),
            "loss_confidence": confidence_loss.detach(),
            "loss_confidence_negative": negative_confidence_loss.detach(),
            "loss_class": class_loss.detach(),
        }

    def _get_class_loss(self, preds, targets, object_exists):
        c = preds[:, : self.num_classes]  # shape (batch, C, S, S)
        c_hat = targets[:, : self.num_classes]  # shape (batch, C, S, S)

        c_loss = self.mse(c_hat, c)  # shape (batch, C, S, S)
        c_loss = c_loss.sum(dim=1)  # shape (batch, S, S)
        c_loss = object_exists * c_loss
        c_loss = c_loss.sum(dim=(1, 2)).mean(dim=0)
        return c_loss

    def _get_confidence_loss(self, preds, ious, responsibility):
        """
        :param preds: shape: (batch, (C + B * 5), S, S)
        :param ious: (batch, B, S, S)
        :param responsibility: (batch, B, S, S)
        :return:
        """
        confidence_indices = [self.num_classes + (i * 5) for i in range(self.num_boxes)]
        c_hat = preds[:, tuple(confidence_indices)]  # (batch, B, S, S)

        c_loss = self.mse(c_hat, ious)  # (batch, B, S, S)
        c_loss = responsibility * c_loss  # (batch, B, S, S)
        c_loss = c_loss.sum(dim=(1, 2, 3)).mean(dim=0)  # scalar

        return c_loss

    def _get_coords_loss(self, preds, targets, responsibility):
        x = targets[:, self.num_classes + 1]  # shape (batch, S, S)
        y = targets[:, self.num_classes + 2]
        w = targets[:, self.num_classes + 3]
        h = targets[:, self.num_classes + 4]
        w_sqrt = torch.sqrt(torch.abs(w))
        h_sqrt = torch.sqrt(torch.abs(h))

        coords_losses = []  # shape (B,
        for i in range(self.num_boxes):
            start = self.num_classes + (i * 5)
            x_hat = preds[:, start + 1]  # shape (batch, S, S)
            y_hat = preds[:, start + 2]
            w_hat = preds[:, start + 3]
            h_hat = preds[:, start + 4]
            w_hat_sqrt = torch.sqrt(torch.abs(w_hat))
            h_hat_sqrt = torch.sqrt(torch.abs(h_hat))

            xy_loss = self.mse(x_hat, x) + self.mse(y_hat, y)
            wh_loss = self.mse(w_hat_sqrt, w_sqrt) + self.mse(h_hat_sqrt, h_sqrt)
            coords_loss = responsibility[:, i] * (xy_loss + wh_loss)
            coords_loss = coords_loss.sum(dim=(1, 2)).mean(
                dim=0
            )  # average over batch, sum over rest.
            coords_losses.append(coords_loss)
        coords_loss = torch.stack(coords_losses).sum(dim=0)  # sum over B
        return coords_loss
