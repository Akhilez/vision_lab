from typing import List, Union
import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class SimpleCNN(nn.Module):
    def __init__(
        self,
        architecture: List[Union[tuple, str, list]],
        in_channels: int,
    ):
        super(SimpleCNN, self).__init__()
        layers = []
        for module in architecture:
            if type(module) is tuple:
                layers.append(self._get_cnn_block(module, in_channels))
                in_channels = module[1]
            elif module == "M":
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=(2, 2),
                        stride=(2, 2),
                    )
                )
            elif type(module) is list:
                for i in range(module[-1]):
                    for j in range(len(module) - 1):
                        layers.append(self._get_cnn_block(module[j], in_channels))
                        in_channels = module[j][1]
        self.model = nn.Sequential(*layers)

    @staticmethod
    def _get_cnn_block(module: tuple, in_channels):
        kernel_size, filters, stride, padding = module
        return CNNBlock(
            in_channels,
            filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        return self.model(x)


"""
Information about architecture config:
- Tuple is structured by (kernel_size, filters, stride, padding)
- "M" is simply maxpooling with stride 2x2 and kernel 2x2
- List is structured by tuples and lastly int with number of repeats
"""

# original_yolo = [
#     (7, 64, 2, 3),
#     "M",
#     (3, 192, 1, 1),
#     "M",
#     (1, 128, 1, 0),
#     (3, 256, 1, 1),
#     (1, 256, 1, 0),
#     (3, 512, 1, 1),
#     "M",
#     [(1, 256, 1, 0), (3, 512, 1, 1), 4],
#     (1, 512, 1, 0),
#     (3, 1024, 1, 1),
#     "M",
#     [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
#     (3, 1024, 1, 1),
#     (3, 1024, 2, 1),
#     (3, 1024, 1, 1),
#     (3, 1024, 1, 1),
# ]
# For pascal voc
# architecture_config = [
#     (7, 64, 2, 3),  # 224
#     "M",  # 112
#     (3, 194, 1, 1),
#     "M",  # 56
#     [(1, 128, 1, 0), (3, 128, 1, 1), 2],
#     "M",  # 28
#     [(1, 128, 1, 0), (3, 128, 1, 1), 4],
#     "M",  # 14
#     [(1, 128, 1, 0), (3, 128, 1, 1), 4],
#     (3, 64, 2, 1),  # 7
#     (3, 32, 1, 1),
# ]
# For mnist aug
architecture_config = [
    (7, 64, 2, 3),  # 112
    "M",  # 56
    (3, 194, 1, 1),
    "M",  # 28
    [(1, 128, 1, 0), (3, 128, 1, 1), 2],
    "M",  # 14
    [(1, 128, 1, 0), (3, 128, 1, 1), 4],
    (3, 32, 1, 0),  # 5
]


class YoloV1(nn.Module):
    def __init__(
        self, split_size: int, num_boxes: int, num_classes: int, in_channels: int
    ):
        super(YoloV1, self).__init__()
        self.backbone = SimpleCNN(architecture_config, in_channels=in_channels)

        S, B, C = split_size, num_boxes, num_classes
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * S * S, 128),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),
            nn.Linear(128, S * S * (C + B * 5)),
        )
        self.final_shape = (-1, (C + B * 5), S, S)

    def forward(self, x):
        x = self.backbone(x)
        out = self.fcs(torch.flatten(x, start_dim=1))
        out = out.view(self.final_shape)
        return out
