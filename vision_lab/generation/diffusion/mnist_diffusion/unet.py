import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class DoubleConv(nn.Module):
    """(Convolution => ReLU) * 2 block with same spatial dimensions."""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv.
    Uses a transposed convolution for upsampling, concatenates with the corresponding encoder feature map.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        """
        Args:
            in_channels: number of channels coming from the previous layer (after upsampling)
            skip_channels: number of channels from the skip connection
            out_channels: number of output channels after the double conv block
        """
        super(Up, self).__init__()
        # Transposed convolution for upsampling: reduce channels to out_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation, the number of input channels is: skip_channels + out_channels.
        self.conv = DoubleConv(skip_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # In case the in/out sizes differ by a few pixels, we pad x
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        # Concatenate along the channel axis
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to map to desired output channels."""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=4, initial_filters=64):
        """
        Configurable U-Net.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB images)
            out_channels (int): Number of output channels (e.g., number of classes)
            depth (int): Number of down/up-sampling layers.
            initial_filters (int): Number of filters in the first convolution layer.
        """
        super(UNet, self).__init__()
        self.depth = depth

        # Encoder: Create a list of filter sizes. Each down layer doubles the number of filters.
        # For depth = 4, for example, the encoder will have channels: [initial_filters, 2*initial_filters, 4*initial_filters, 8*initial_filters, 16*initial_filters]
        self.enc_channels = [initial_filters * (2 ** i) for i in range(depth + 1)]

        # Initial convolution block (first level of encoder)
        self.inc = DoubleConv(in_channels, self.enc_channels[0])

        # Down-sampling layers
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(Down(self.enc_channels[i], self.enc_channels[i + 1]))

        # Up-sampling layers: reverse the encoder channel order (excluding the bottom-most layer)
        self.ups = nn.ModuleList()
        for i in range(depth, 0, -1):
            # For the up block:
            #   in_channels comes from the previous layer (which is enc_channels[i])
            #   skip_channels is from the encoder (enc_channels[i-1])
            #   out_channels is chosen to be enc_channels[i-1]
            self.ups.append(Up(in_channels=self.enc_channels[i],
                               skip_channels=self.enc_channels[i - 1],
                               out_channels=self.enc_channels[i - 1]))

        # Final 1x1 convolution layer to produce the desired output channels.
        self.outc = OutConv(self.enc_channels[0], out_channels)

    def forward(self, x):
        # Encoder: store the outputs for skip connections.
        x_enc = []
        x = self.inc(x)
        x_enc.append(x)
        for down in self.downs:
            x = down(x)
            x_enc.append(x)

        # x_enc[-1] is the output at the bottom of the U-Net.
        # Decoder: reverse the list (skip connections) except for the bottom element.
        x = x_enc[-1]
        for i, up in enumerate(self.ups):
            # The corresponding skip connection is from x_enc[-(i + 2)]
            skip = x_enc[-(i + 2)]
            x = up(x, skip)

        logits = self.outc(x)
        return logits

    @classmethod
    def encode_t(cls, x, t):
        # x: (B, C, H, W)
        # t: (B, 1)
        # output: (B, C+1, H, W)
        t = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        return torch.cat([x, t], dim=1)


# Example usage:
if __name__ == "__main__":
    # Create a model instance with configurable depth and initial filters.
    # For example, depth=4 and initial_filters=64.
    model = UNet(in_channels=3, out_channels=3, depth=3, initial_filters=16)

    # Create a random input tensor with batch size 1, 3 channels, and 256x256 spatial dimensions.
    x = torch.randn(1, 3, 256, 256)

    summary(model, x.shape[1:], device="cpu")

    # Forward pass
    output = model(x)
    print("Output shape:", output.shape)  # Expected shape: [1, 1, 256, 256]
