from typing import Any
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import AverageMeter, MetricCollection
from torchsummary import summary
from object_detection.yolo1.datasets.voc_dataset import VocYoloDataModule
from object_detection.yolo1.loss import YoloV1Loss
from object_detection.yolo1.model import YoloV1
from settings import BASE_DIR


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


class YoloV1PL(pl.LightningModule):
    def __init__(
        self,
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
        self.hp = hp
        self.yolo_v1 = YoloV1(
            split_size=grid_size,
            num_boxes=num_boxes,
            num_classes=num_classes,
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
        losses = self.criterion(preds, targets)
        return {
            "images": images,
            "targets": targets,
            "preds": preds.detach(),
            "losses": losses,
            "batch_index": batch_index,
        }

    def training_step_end(self, outputs: dict):
        losses = outputs["losses"]
        batch_index = outputs["batch_index"]
        images = outputs["images"]

        self.metrics_train.update_each(losses)
        self.log("train/loss_step", losses["loss"], prog_bar=True)
        if batch_index == 0:
            images_to_log = images[: self.hp["num_log_images"]]
            self.logger.experiment.log(
                {"train/predictions": wandb.Image(images_to_log)}
            )

    def training_epoch_end(self, outputs):
        self.log_dict(self.metrics_train.compute())

    def validation_step(self, batch, _index):
        images, targets = batch
        preds = self(images)
        losses = self.criterion(preds, targets)
        return losses

    def validation_step_end(self, losses):
        self.metrics_val.update_each(losses)

    def on_validation_epoch_end(self):
        self.log_dict(self.metrics_val.compute())
