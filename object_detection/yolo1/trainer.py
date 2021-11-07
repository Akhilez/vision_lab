from os.path import join
import pytorch_lightning as pl
import torch
import yaml
from torchmetrics import AverageMeter
from object_detection.yolo1.datasets.transforms import transform_targets_from_yolo
from object_detection.yolo1.experiment_logger import get_wandb_visualizations
from object_detection.yolo1.loss import YoloV1Loss
from object_detection.yolo1.metrics import MyMetricCollection, get_ious
from object_detection.yolo1.model import YoloV1


class YoloV1PL(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_boxes: int,
        num_classes: int,
        grid_size: int,
        image_height: int,
        image_width: int,
        lambda_coord: float,
        lambda_object_exists: float,
        lambda_no_object: float,
        lambda_class: float,
        **hp,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.hp = hp
        self.yolo_v1 = YoloV1(
            split_size=grid_size,
            num_boxes=num_boxes,
            num_classes=num_classes,
            in_channels=in_channels,
        )
        self.criterion = YoloV1Loss(
            num_boxes=num_boxes,
            num_classes=num_classes,
            image_height=image_height,
            image_width=image_width,
            lambda_coord=lambda_coord,
            lambda_object_exists=lambda_object_exists,
            lambda_no_object=lambda_no_object,
            lambda_class=lambda_class,
        )

        # --- metrics ---
        self.metrics_train = MyMetricCollection(
            {
                "loss": AverageMeter(),
                "loss_coords": AverageMeter(),
                "loss_confidence": AverageMeter(),
                "loss_confidence_negative": AverageMeter(),
                "loss_class": AverageMeter(),
            },
            prefix="train/",
        )
        self.metrics_val = self.metrics_train.clone(prefix="val/")

    def forward(self, x):
        return self.yolo_v1(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hp["lr_initial"])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hp["lr_decay_every"],
            gamma=self.hp["lr_decay_by"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
            },
        }

    def training_step(self, batch, batch_index):
        images, targets = batch
        preds = self(images)

        preds_denorm, targets_denorm = self._denorm(preds.detach(), targets)
        ious = get_ious(preds_denorm, targets_denorm)  # shape: (batch, B, S, S)

        losses = self.criterion(preds, targets, ious)

        # ----------- metrics and logs -------------
        self.metrics_train.update_each(losses)
        self.log("train/loss_step", losses["loss"])
        if batch_index == 0:
            self._log_visualizations(
                "train/overlays",
                images,
                preds_denorm,
                targets_denorm,
                preds,
                targets,
                ious,
            )

        return losses["loss"]

    def training_epoch_end(self, outputs):
        self.log_dict(self.metrics_train.compute())

    def validation_step(self, batch, _index):
        images, targets = batch
        preds = self(images)

        preds_denorm, targets_denorm = self._denorm(preds.detach(), targets)
        ious = get_ious(preds_denorm, targets_denorm)  # shape: (batch, B, S, S)

        losses = self.criterion(preds, targets, ious)

        return losses

    def validation_step_end(self, losses):
        self.metrics_val.update_each(losses)

    def on_validation_epoch_end(self):
        self.log_dict(self.metrics_val.compute())

    def _denorm(self, preds, targets):

        # shape: (batch, 4, B, S, S)
        preds_denorm = transform_targets_from_yolo(
            preds, self.image_height, self.image_width, self.num_classes
        )

        # shape: (batch, 4, 1, S, S)
        targets_denorm = transform_targets_from_yolo(
            targets, self.image_height, self.image_width, self.num_classes
        )

        return preds_denorm, targets_denorm

    def _log_visualizations(
        self, key, images, preds_denorm, targets_denorm, preds, targets, ious
    ):
        confidence_indices = [self.num_classes + (i * 5) for i in range(self.num_boxes)]
        confidences = preds[:, tuple(confidence_indices)]  # (batch, B, S, S)

        classes_true = targets[:, : self.num_classes]  # shape (batch, C, S, S)
        classes_pred = preds[:, : self.num_classes]  # shape (batch, C, S, S)

        object_exists = targets[:, self.num_classes]  # shape: (batch, S, S)

        log = get_wandb_visualizations(
            images,
            preds_denorm,
            targets_denorm,
            object_exists,
            classes_true,
            classes_pred,
            ious,
            confidences,
            confidence_threshold=0.1,
            limit=self.hp["num_log_images"],
        )

        self.logger.experiment.log({key: log})


def save_final_results(output_path, wandb_logger, checkpoint_callback):
    checkpoint_path = checkpoint_callback.best_model_path
    wandb_logger.experiment.save(checkpoint_path)

    summary = dict(wandb_logger.experiment.summary)
    del summary["train/overlays"]

    final_dict = {
        "config": dict(wandb_logger.experiment.config),
        "summary": summary,
        "best_checkpoint_path": checkpoint_path,
    }

    config_path = join(output_path, "output.yaml")
    with open(config_path, "w") as output_file:
        yaml.safe_dump(final_dict, output_file)
    wandb_logger.experiment.save(config_path)
