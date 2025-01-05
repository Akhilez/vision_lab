from os.path import join
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchsummary import summary
from object_detection.yolo1.datasets.mnist_aug_dataset import MnistAugDataModule
from object_detection.yolo1.trainer import YoloV1PL, save_final_results
from settings import BASE_DIR, device


def train_mnist_aug_yolo1(hp, config):
    output_path = join(config["output_path"], config["project_name"])
    data_module = MnistAugDataModule(**config, **hp)
    model = YoloV1PL(**hp, **config).to(device).float()
    summary(
        model, (config["in_channels"], config["image_height"], config["image_width"])
    )
    wandb_logger = WandbLogger(
        project=config["project_name"],
        log_model=False,
        dir=join(output_path, "wandb"),
        config={**hp, **config},
        mode=config['wandb_mode'],
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=join(output_path, "checkpoints"),
        filename=config["project_name"] + "__{epoch:02d}",
        save_top_k=3,
        mode="min",
        auto_insert_metric_name=True,
    )
    trainer = pl.Trainer(
        gpus=config["num_gpus"],
        max_epochs=hp["epochs"],
        default_root_dir=output_path,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    wandb_logger.watch(model)

    trainer.fit(model, datamodule=data_module)

    save_final_results(output_path, wandb_logger, checkpoint_callback)


if __name__ == "__main__":
    architecture_config = [
        # (kernel_size, filters, stride, padding)
        (7, 64, 2, 3),  # 112 -> 56
        "M",  # 56 -> 28
        (3, 194, 1, 1),
        # [(1, 128, 1, 0), (3, 128, 1, 1), 2],
        "M",  # 28 -> 14
        [(1, 128, 1, 0), (3, 128, 1, 1), 1],
        "M",  # 14 -> 7
        [(1, 128, 1, 0), (3, 128, 1, 1), 1],
        (3, 32, 1, 0),  # 7 -> 5
    ]
    hp = {
        "epochs": 2,
        "batch_size": 2,
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
        "project_name": "mnist_detection_yolo1",
        "output_path": f"{BASE_DIR}/object_detection/yolo1/output",
        "data_path_train": f"{BASE_DIR}/data/mnist_detection_10k/train",
        "data_path_val": f"{BASE_DIR}/data/mnist_detection_10k/test",
        "in_channels": 1,
        "num_classes": 10,
        "image_height": 112,
        "image_width": 112,
        "dataset_size": None,
        "num_log_images": 3,
        "dataloader_num_workers": 0,
        "num_gpus": 0,
        "architecture": architecture_config,
        "wandb_mode": "disabled",
    }
    train_mnist_aug_yolo1(hp, config)
