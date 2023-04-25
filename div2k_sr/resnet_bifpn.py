import torch.nn as nn
from torch.nn import functional as F
import torch
import torchvision.transforms as transforms
from PIL import Image

from bifpn import BiFPN
from resnet import ResNet


def inference(model, lr_img: Image.Image, hr_size: tuple[int, int]):
    channels = lr_img.convert("YCbCr").split()
    pred_channel_y = model(channels[0])
    transform = transforms.Resize(size=hr_size)
    interpolated_img = transform(lr_img)
    interpolated_channels = list(interpolated_img.convert("YCbCr").split())
    interpolated_channels[0] = pred_channel_y
    pred_hr_img = Image.merge('YCbCr', tuple(interpolated_channels))
    return pred_hr_img.convert("RGB")


class UpsampleCat(nn.Module):
    def __init__(self):
        super(UpsampleCat, self).__init__()

    def forward(self, x):
        """Upsample and concatenate feature maps."""
        assert isinstance(x, list) or isinstance(x, tuple)
        x0 = x[0]
        _, _, H, W = x0.size()
        for i in range(1, len(x)):
            x0 = torch.cat([x0, F.interpolate(x[i], (H, W))], dim=1)
        return x0


class ResNetBiFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fpn_num_filters = 128
        self.fpn_cell_repeats = 3
        conv_channel_coef = [128, 256, 512]

        self.backbone_net = ResNet()

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters,
                    conv_channel_coef,
                    True if _ == 0 else False,
                    attention=True,
                    use_p8=False)
              for _ in range(self.fpn_cell_repeats)])
        self.up = UpsampleCat()
        self.last_conv = nn.Conv2d(640, 256, 1)
        self.pixel_shuffle = nn.PixelShuffle(16)

    def forward(self, inputs):
        x3, x4, x5 = self.backbone_net.get_features(inputs)
        features = (x3, x4, x5)
        x1, x2, x3, x4, x5 = self.bifpn(features)
        x = self.up((x1, x2, x3, x4, x5))
        x = self.last_conv(x)
        x = self.pixel_shuffle(x)
        return x


if __name__ == "__main__":
    model = ResNetBiFPN()
    data = torch.ones((2, 1, 128, 128))
    out = model(data)
    print(data.shape, out.shape)
