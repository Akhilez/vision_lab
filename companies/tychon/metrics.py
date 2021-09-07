from copy import deepcopy
import torch


class SegmentationMetrics:
    """
    This class keeps track of all the metrics during training and evaluation.
    """
    def __init__(self, num_classes):
        self.metrics_to_track = [
            'accuracy',
            'iou',
        ]
        self.epoch_metrics = {key: 0.0 for key in self.metrics_to_track}
        self.n_batches = 0
        self.num_classes = num_classes

    def step_batch(
        self,
        masks_pred: torch.Tensor,
        masks_true: torch.Tensor,
        **other,
    ):
        """
        masks_pred: (batch_size, num_classes, H, W)
        masks_true: (batch_size, H, W)
        """
        batch_metrics = other

        masks_pred = masks_pred.argmax(dim=1).detach().cpu()
        masks_true = masks_true.cpu()

        # Finding value for each metric
        hist = torch.zeros((self.num_classes, self.num_classes))
        for t, p in zip(masks_true, masks_pred):
            hist += fast_hist(t.flatten(), p.flatten(), self.num_classes)

        batch_metrics['accuracy'] = float(overall_pixel_accuracy(hist))
        batch_metrics['iou'] = float(jaccard_index(hist))

        self._log_batch(batch_metrics)
        return batch_metrics

    def step_epoch(self) -> dict:
        for key in self.epoch_metrics:
            self.epoch_metrics[key] /= self.n_batches
            metrics = deepcopy(self.epoch_metrics)
            self.clear()
            return metrics

    def _log_batch(self, batch_metrics: dict):
        for key in batch_metrics:
            if key not in self.epoch_metrics:
                self.epoch_metrics[key] = 0
            self.epoch_metrics[key] += float(batch_metrics[key])
        self.n_batches += 1

    def clear(self):
        self.epoch_metrics = {key: 0.0 for key in self.metrics_to_track}
        self.n_batches = 0


"""
Common image segmentation metrics.
Source: https://github.com/kevinzakka/pytorch-goodies/blob/master/metrics.py
"""

EPS = 1e-10


def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])


def fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist


def overall_pixel_accuracy(hist):
    """Computes the total pixel accuracy.

    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.

    Args:
        hist: confusion matrix.

    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc


def jaccard_index(hist):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).

    Args:
        hist: confusion matrix.

    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)
    return avg_jacc
