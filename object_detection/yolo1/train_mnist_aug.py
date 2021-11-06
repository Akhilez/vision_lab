import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchsummary import summary
from object_detection.yolo1.datasets.mnist_aug_dataset import MnistAugDataModule
from object_detection.yolo1.trainer import YoloV1PL
from settings import BASE_DIR, device

hp = {
    "epochs": 100,
    "batch_size": 32,
    "lr_initial": 0.0001,
    "lr_decay_every": 20,
    "lr_decay_by": 0.99,
    "grid_size": 5,
    "data_augment": False,
    "num_boxes": 2,
    "lambda_coord": 5,
    "lambda_object_exists": 1,
    "lambda_no_object": 0.5,
    "lambda_class": 1,
}

config = {
    "output_path": f"{BASE_DIR}/object_detection/yolo1/output",
    "data_path_train": f"{BASE_DIR}/data/mnist_detection/sample/train",
    "data_path_val": f"{BASE_DIR}/data/mnist_detection/sample/test",
    "in_channels": 1,
    "num_classes": 10,
    "image_height": 112,
    "image_width": 112,
    "dataset_size": None,
    "num_log_images": 3,
    "dataloader_num_workers": 0,
    "num_gpus": 0,
}

data_module = MnistAugDataModule(**config, **hp)
model = YoloV1PL(**hp, **config).to(device).float()
summary(model, (config["in_channels"], config["image_height"], config["image_width"]))
wandb_logger = WandbLogger(
    project="mnist_detection_yolo1", log_model=False, save_dir="../wandb_logs"
)
trainer = pl.Trainer(
    gpus=config["num_gpus"],
    max_epochs=hp["epochs"],
    default_root_dir=config["output_path"],
    logger=wandb_logger,
)
# wandb_logger.watch(model)

trainer.fit(model, datamodule=data_module)
