from unittest import TestCase
import torch
from object_detection.yolo1.loss import YoloV1Loss
from object_detection.yolo1.transforms import YoloV1Transforms


class TestYolo(TestCase):
    def setUp(self) -> None:
        self.h = 12
        self.w = 12
        self.num_classes = 2
        self.grid_size = 3
        self.num_boxes = 2
        self.batch_size = 2

        self.transforms = YoloV1Transforms(
            h=self.h,
            w=self.w,
            augment=False,
            num_classes=self.num_classes,
            grid_size=self.grid_size,
        )

        # ---- TARGETS ----

        self.x1y1x2y2 = [[3, 1, 5, 5], [5, 6, 9, 8]]
        self.cxcywh = [[4, 3, 3, 5], [7, 7, 5, 3]]
        self.responsible_cells = [[0, 1], [1, 1]]
        self.classes = [0, 1]
        self.yolo_boxes = [[0 / 4, 3 / 4, 1 / 4, 5 / 12], [3 / 4, 3 / 4, 5 / 12, 1 / 4]]

        self.targets = torch.zeros(
            (self.batch_size, self.num_classes + 5, self.grid_size, self.grid_size)
        )

        self.targets[0, :, 0, 1] = torch.tensor([1, 0, 1.0] + self.yolo_boxes[0])
        self.targets[0, :, 1, 1] = torch.tensor([0, 1, 1.0] + self.yolo_boxes[1])
        self.targets[1] = self.targets[0]

        # ---- PREDICTIONS ------

        self.pred_classes = [[0.1, 0.8], [0.9, 0.2], [0.4, 0.5]]
        self.pred_cells = [[0, 1], [0, 2], [1, 1]]
        self.pred_confidences = [[0.6, 0.3], [0.8, 0.8], [0.7, 0.6]]
        self.pred_yolo_format = [
            [[1 / 4, 2 / 4, 3 / 12, 5 / 12], [1 / 4, 3 / 4, 5 / 12, 3 / 12]],
            [[1 / 4, 2 / 4, 3 / 12, 3 / 12], [2 / 4, 1 / 4, 3 / 12, 3 / 12]],
            [[2 / 4, 3 / 4, 3 / 12, 3 / 12], [3 / 4, 2 / 4, 3 / 12, 3 / 12]],
        ]
        self.pred_x1y1x2y2 = [
            [[4, 0, 6, 4], [3, 2, 7, 4]],
            [[8, 1, 10, 3], [9, 0, 11, 2]],
            [[5, 6, 7, 8], [6, 5, 8, 7]],
        ]
        self.ious = [[4 / 11, 9 / 12], [0, 0], [3 / 5, 6 / 18]]

        self.preds = torch.zeros(
            (
                self.batch_size,
                self.num_classes + (5 * self.num_boxes),
                self.grid_size,
                self.grid_size,
            )
        )
        for i in range(3):
            self.preds[
                0, :, self.pred_cells[i][0], self.pred_cells[i][1]
            ] = torch.tensor(
                [
                    self.pred_classes[i]
                    + [self.pred_confidences[i][0]]
                    + self.pred_yolo_format[i][0]
                    + [self.pred_confidences[i][1]]
                    + self.pred_yolo_format[i][1]
                ]
            )
        self.preds[1] = self.preds[0]

        # -------- LOSSES ----------

        self.criterion = YoloV1Loss(
            num_boxes=self.num_boxes,
            num_classes=self.num_classes,
            image_height=self.h,
            image_width=self.w,
            lambda_coord=5,
            lambda_object_exists=1,
            lambda_no_object=0.5,
            lambda_class=1,
        )

    def test_labels_to_targets(self):
        targets = self.transforms.transform_targets_to_yolo(self.x1y1x2y2, self.classes)
        assert torch.all(targets == self.targets[0])

    def test_yolo_to_normal_2(self):
        preds = self.transforms.transform_targets_from_yolo(
            self.preds,
            image_height=self.h,
            image_width=self.w,
            num_classes=self.num_classes,
        )

        assert list(preds.shape) == [
            self.batch_size,
            4,
            self.num_classes,
            self.grid_size,
            self.grid_size,
        ]

        preds_exact = []
        for cell in self.pred_cells:
            boxes = preds[0, :, :, cell[0], cell[1]]
            preds_exact.append(boxes.T.tolist())

        assert self.pred_x1y1x2y2 == preds_exact

    def test_yolo_to_normal(self):
        h, w = 448, 448
        grid_rows, grid_cols = 7, 7
        cell_h = h / grid_rows
        cell_w = w / grid_cols

        assert cell_h == 64.0
        assert cell_w == 64.0
        print(cell_h, cell_w)

        cols = (
            torch.arange(start=0, end=w, step=cell_w)
            .view((1, grid_cols))
            .expand((grid_rows, grid_cols))
        )
        rows = (
            torch.arange(start=0, end=h, step=cell_h)
            .view((grid_rows, 1))
            .expand((grid_rows, grid_cols))
        )
        origins = torch.stack((rows, cols))
        print(
            origins,
            origins.shape,
        )

        num_boxes = 2
        num_classes = 20

        idx = [], [], [], []
        idx_x, idx_y, idx_w, idx_h = idx

        for i in range(num_boxes):
            for j in range(4):
                idx[j].append(num_classes + (i * 5) + j + 1)

        print(idx)

        x = 19 + 1 + 1
        y = x + 1
        w = y + 1
        h = w + 1
        x2 = h + 1 + 1
        y2 = x2 + 1
        w2 = y2 + 1
        h2 = w2 + 1

        true = ([x, x2], [y, y2], [w, w2], [h, h2])
        print(true)

        assert idx == true

    def test_transform_targets_from_yolo(self):
        # shape (batch, C+5B, S, S)
        # shape (2, 2 + 5*2, 3, 3)
        sample_prediction = []

    def test_loss(self):
        import math

        l_coords = 1 / 8 + 2 - math.sqrt(15) / 2
        l_obj = 13 / 490
        l_no_obj = 2.0
        l_class = 1.86
        expected_loss = 5 * l_coords + l_obj + l_class + 0.5 * l_no_obj
        found_loss = self.criterion(self.preds, self.targets)

        print("loss ", expected_loss, float(found_loss["loss"]))
        print("loss_coords ", l_coords, float(found_loss["loss_coords"]))
        print("loss_confidence ", l_obj, float(found_loss["loss_confidence"]))
        print(
            "loss_confidence_negative ",
            l_no_obj,
            float(found_loss["loss_confidence_negative"]),
        )
        print("loss_class ", l_class, float(found_loss["loss_class"]))

        """
        5.9290722467263555
        {
            'loss': tensor(10.2864),
            'loss_coords': tensor(1.6047),
            'loss_confidence': tensor(0.1730),
            'loss_confidence_negative': tensor(0.3200),
            'loss_class': tensor(1.9300)
        }
        """

    def test_iou(self):
        ious = self.criterion._get_ious(
            self.preds,
            self.targets,
            image_height=self.h,
            image_width=self.w,
        )
        print(ious[0])
        print(9 / 21, 3 / 5)

        assert ious[0, 1, 0, 1] == 9 / 21
        assert ious[0, 0, 1, 1] == 3 / 5
