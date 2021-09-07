from torch import nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class PaintingModel(nn.Module):
    def __init__(self):
        super(PaintingModel, self).__init__()
        self.backbone = deeplabv3_mobilenet_v3_large(pretrained=True, progress=True, num_classes=21)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(21, 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )

    def forward(self, x):
        x = self.backbone(x)['out']
        x = self.classifier(x)
        return x
