from os import makedirs
from os.path import join
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn import functional as F
from model import PaintingModel
from metrics import SegmentationMetrics
from dataset import load_image_paths, PaintingsDataset
from torchvision import transforms

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


def _save_model(model, cfg: dict, postfix=None):
    postfix = f"__{postfix}" if postfix is not None else ""
    file_name = f"model{postfix}.pth"
    checkpoint_path = join(cfg['output_path'], "checkpoints", file_name)
    torch.save(model.state_dict(), checkpoint_path)


def evaluate_dataset(model, data_loader, metrics, seg_criterion):
    model.eval()
    for images, masks in data_loader:
        images = images.to(device).float()
        masks = masks.to(device).long()

        with torch.no_grad():
            pred_masks = model(images)

        loss = seg_criterion(pred_masks, masks)

        metrics.step_batch(
            masks_true=masks,
            masks_pred=pred_masks,
            loss=float(loss),
        )
    scores = metrics.step_epoch()
    return scores


def _cross_entropy_loss():
    criterion = nn.CrossEntropyLoss()

    def loss_fn(pred, true):
        pred = F.softmax(pred, dim=1)
        return criterion(pred, true)

    return loss_fn


def log_metrics(logger, scores: dict, prefix: str, step):
    for key in scores:
        logger.add_scalar(f'{prefix}/{key}', scores[key], step)


def preprocess_pretrained():
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess


# ----------- MAIN --------------


def main():
    cfg = {
        'data_path': './data',
        'valset_size': 0.1,
        'epochs': 10,
        'batch_size': 2,
        'learning_rate': {
            'initial': 0.0001,
            'decay_every': 30,
            'decay_by': 0.3,
        },
        'output_path': './output'
    }
    makedirs(join(cfg['output_path'], "checkpoints"), exist_ok=True)
    logger = SummaryWriter()
    metrics = SegmentationMetrics(num_classes=2)

    # -------------- model ---------------

    model = PaintingModel()

    if cfg.get('checkpoint'):
        model.load(cfg['checkpoint'])
    model = model.float().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['learning_rate']['initial']
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg['learning_rate']['decay_every'],
        gamma=cfg['learning_rate']['decay_by'],
    )

    seg_criterion = _cross_entropy_loss()

    # ------------- Data -------------

    image_paths = load_image_paths([cfg['data_path']])
    train_images, val_images = train_test_split(
        image_paths, test_size=cfg['valset_size'], shuffle=True
    )
    train_loader = DataLoader(
        PaintingsDataset(train_images, preprocess_pretrained()),
        batch_size=cfg['batch_size'],
        num_workers=0,
        shuffle=True,
    )
    val_loader = DataLoader(
        PaintingsDataset(val_images, preprocess_pretrained()),
        batch_size=cfg['batch_size'],
        num_workers=0,
        shuffle=False,
    )

    # ------------- training loop --------------

    try:
        for epoch in range(cfg['epochs']):
            model.train()
            with tqdm(train_loader, unit="batch") as batches_bar:
                batches_bar.set_description(f"Epoch {epoch}")
                for images, masks in batches_bar:
                    images = images.to(device).float()
                    masks = masks.to(device).long()

                    pred_masks = model(images)

                    loss = seg_criterion(pred_masks, masks)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # -------- Metrics ------------

                    batch_metrics = metrics.step_batch(
                        masks_true=masks,
                        masks_pred=pred_masks,
                        loss=float(loss),
                        learning_rate=float(lr_scheduler.get_last_lr()[0]),
                    )
                    batches_bar.set_postfix(
                        loss=batch_metrics['loss'],
                        iou=batch_metrics['iou']
                    )

            lr_scheduler.step()
            if (epoch + 1) % cfg['model_save_frequency'] == 0:
                _save_model(model, cfg, epoch)

            scores = metrics.step_epoch()
            log_metrics(logger, scores, 'train', epoch)

            val_scores = evaluate_dataset(model, val_loader, metrics, seg_criterion)
            log_metrics(logger, val_scores, 'val', epoch)

    except KeyboardInterrupt:
        print("Stopping early.")
        optimizer.zero_grad()

    _save_model(model, cfg)
    val_scores = evaluate_dataset(model, val_loader, metrics, seg_criterion)
    logger.add_hparams(cfg, val_scores)


if __name__ == "__main__":
    main()
