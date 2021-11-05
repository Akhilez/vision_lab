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
    confidences: torch.Tensor,
    limit: int = 1,
) -> wandb.Image:
    """
    Okay, so
    :param images: torch.Tensor
    :param predictions: de=normalized. Shape: (batch, 4, B, S, S)
    :param targets: de-normalized. Shape: (batch, 4, 1, S, S)
    :param confidences: A 0-1 tensor of shape (batch, B)
    :param limit: int
    :return:
    """
    for i in range(limit):
        visualization = {
            "predictions": {
                "box_data": [
                    {
                        "position": {
                            "minX": 0,
                            "minY": 0,
                            "maxX": 0,
                            "maxY": 0,
                        },
                        "domain": "pixel",
                        "class_id": 1,
                        "scores": {"iou": 0.9},
                    }
                ],
            },
            "ground_truth": {},
        }
