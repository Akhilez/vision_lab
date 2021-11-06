"""
box_data: (list of dictionaries) One dictionary for each bounding box, containing:
    position: (dictionary) the position and size of the bounding box, in one of two formats
        Note that boxes need not all use the same format.
        {"minX", "minY", "maxX", "maxY"}: (dictionary) A set of coordinates defining
            the upper and lower bounds of the box (the bottom left and top right corners)
        {"middle", "width", "height"}: (dictionary) A set of coordinates defining the
            center and dimensions of the box, with "middle" as a list [x, y] for the
            center point and "width" and "height" as numbers
    domain: (string) One of two options for the bounding box coordinate domain
        null: By default, or if no argument is passed, the coordinate domain
            is assumed to be relative to the original image, expressing this box as a fraction
            or percentage of the original image. This means all coordinates and dimensions
            passed into the "position" argument are floating point numbers between 0 and 1.
        "pixel": (string literal) The coordinate domain is set to the pixel space. This means all
            coordinates and dimensions passed into "position" are integers within the bounds
            of the image dimensions.
    class_id: (integer) The class label id for this box
    scores: (dictionary of string to number, optional) A mapping of named fields
            to numerical values (float or int), can be used for filtering boxes in the UI
            based on a range of values for the corresponding field
    box_caption: (string, optional) A string to be displayed as the label text above this
            box in the UI, often composed of the class label, class name, and/or scores

class_labels: (dictionary, optional) A map of integer class labels to their readable class names

"""
import torch
import wandb


def get_wandb_visualizations(
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    object_exists: torch.Tensor,
    classes_true: torch.Tensor,
    classes_pred: torch.Tensor,
    ious: torch.Tensor,
    confidences: torch.Tensor,
    confidence_threshold: float = 0.5,
    limit: int = 1,
) -> wandb.Image:
    """
    Wait, there are many boxes in predictions and targets. Not all of them will be logged.
    How to filter?
    Predictions:
        Find all the pred_boxes where confidence > threshold.
    Targets:
        Find the ones where object_exists channel is 1.
    :param images: torch.Tensor
    :param predictions: de-normalized. Shape: (batch, 4, B, S, S)
    :param targets: de-normalized. Shape: (batch, 4, 1, S, S)
    :param object_exists: shape: (batch, S, S)
    :param classes_true: tensor of shape (batch, C, S, S)
    :param classes_pred: tensor of shape (batch, C, S, S)
    :param ious:  shape: (batch, B, S, S)
    :param confidences: A 0-1 tensor of shape (batch, B, S, S)
    :param confidence_threshold: float value to filter best boxes.
    :param limit: int
    :return:
    """
    classes_true = classes_true.argmax(dim=1)  # shape (batch, S, S)
    classes_pred = classes_pred.argmax(dim=1)  # shape (batch, S, S)
    batch_size, num_boxes, grid_size, _ = confidences.shape

    for b in range(limit):

        pred_boxes = []
        true_boxes = []

        for cell_box in range(num_boxes):
            # shape: (n, 2)
            filtered_indices_pred = (
                confidences[b, cell_box] > confidence_threshold
            ).nonzero()
            for row, col in filtered_indices_pred:
                min_x, min_y, max_x, max_y = predictions[b, :, cell_box, row, col]
                confidence = confidences[b, cell_box, row, col]
                iou = ious[b, cell_box, row, col]
                class_pred = classes_pred[b, row, col]
                pred_boxes.append(
                    {
                        "position": {
                            "minX": float(min_y),
                            "minY": float(min_x),
                            "maxX": float(max_y),
                            "maxY": float(max_x),
                        },
                        "domain": "pixel",
                        "class_id": int(class_pred),
                        "scores": {"confidence": float(confidence), "iou": float(iou)},
                    }
                )

        if len(pred_boxes) == 0:
            continue

        # shape: (n, 2)
        filtered_indices_true = (object_exists[b] > 0).nonzero()
        for row, col in filtered_indices_true:
            min_x, min_y, max_x, max_y = targets[b, :, 0, row, col]
            class_true = classes_true[b, row, col]
            true_boxes.append(
                {
                    "position": {
                        "minX": float(min_y),
                        "minY": float(min_x),
                        "maxX": float(max_y),
                        "maxY": float(max_x),
                    },
                    "domain": "pixel",
                    "class_id": int(class_true),
                }
            )

        return wandb.Image(
            images[b],
            boxes={
                "predictions": {"box_data": pred_boxes},
                "ground_truth": {"box_data": true_boxes},
            },
        )
