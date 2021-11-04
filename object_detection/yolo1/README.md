# YOLO v1

YOLO = You Only Look Once

TODO:

- [ ] metrics
- [ ] viz logging

[paper](https://arxiv.org/pdf/1506.02640.pdf)

Key points:

- S x S grid. S = 7
- predicts B boxes for each cell. B = 2
- Responsible cell:
    - the cell that contains bbox midpoint.
    - Among B predicted boxes, only the one that has highest IoU will be responsible.
- predicts confidence each cell. confidence = IoU
- predicts x, y, w, h each cell:
    - x, y: they are midpoint coordinates relative to cell origin, h, w.
        Meaning, cell h, w are 1, 1, and x, y will be in [0, 1]
    - h, w: they are bbox height, width relative to whole image.
- predicts C classes each cell.
- All are trained only when the cell is responsible for a bbox.
- Each cell can only predict 1 object. although it tries to predict B bboxes
- Predicted tensor is of shape [S, S, (C + 5B)]
- Architecture is simply a CNN followed by a flatten and fully-connected layers.
- While inference, multiply C probabilities with predicted confidence.
- While inference, apply NMS
- All losses are MSE variations.

Hyperparams:

- leaky relu
- batch size 64
- epochs 135 (with pre-trained)
- momentum 0.9
- decay: 0.0005
- lr:
    - 10^-3 for few epochs.
    - 10^-2 for +75 epochs
    - 10^-3 for +30 epochs.
    - 10^-4 for +30 epochs.
- Extensive augmentation:
    - Random scaling and translation up to 20%
    - randomly adjust the exposure and saturation of the image by up to a factor of 1.5 in the HSV color space.
- dropout of 0.5 on last fully-connected

Losses:

- Object exists & responsible: lambda_coord * sum((x - xhat)^2 + (y - yhat)^2)
- Object exists & responsible: lambda_coord * sum((sqrt(w) - sqrt(w_hat))^2 + (sqrt(h) - sqrt(h_hat))^2)
- Object exists & responsible: 1 * sum((confidence - confidence_hat)^2)
- No-object exists & responsible: lambda_no_object * sum((confidence - confidence_hat)^2)
- Object exists: sum((probability(c) - probability(c_hat))^2)

confidence = IoU
lambda_coord = 5
lambda_no_object = 0.5
