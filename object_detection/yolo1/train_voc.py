import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchsummary import summary
from object_detection.yolo1.datasets.voc_dataset import VocYoloDataModule
from object_detection.yolo1.trainer import YoloV1PL
from settings import BASE_DIR

hp = {
    "epochs": 100,
    "batch_size": 32,
    "lr_initial": 0.0001,
    "lr_decay_every": 20,
    "lr_decay_by": 0.99,
    "grid_size": 7,
    "data_augment": True,
    "num_boxes": 2,
    "lambda_coord": 5,
    "lambda_object_exists": 1,
    "lambda_no_object": 0.5,
    "lambda_class": 1,
}

config = {
    "output_path": f"{BASE_DIR}/object_detection/yolo1/output",
    "val_split": 0.1,
    "data_path": f"{BASE_DIR}/object_detection/data",
    "in_channels": 3,
    "num_classes": 20,
    "image_height": 448,
    "image_width": 448,
    "dataset_size": None,
    "num_log_images": 3,
    "dataloader_num_workers": 4,
    "num_gpus": 1,
}

data_module = VocYoloDataModule(**config, **hp)
model = YoloV1PL(**hp, **config).cuda().float()
summary(model, (3, config["image_height"], config["image_width"]))
wandb_logger = WandbLogger(
    project="yolo_test", log_model=False, save_dir="../wandb_logs"
)
trainer = pl.Trainer(
    gpus=config["num_gpus"],
    max_epochs=hp["epochs"],
    default_root_dir=config["output_path"],
    logger=wandb_logger,
)
# wandb_logger.watch(model)

trainer.fit(model, datamodule=data_module)
