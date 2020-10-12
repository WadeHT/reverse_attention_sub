import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F

"""define Unet"""


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, image_size=224, pooling=True, input_channel=1, bilinear=True, output_channel=1, **kargs):
        super(Unet, self).__init__()
        self.n_channels = input_channel
        self.n_classes = 1  # n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, output_channel)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def name(self):
        return "Unet"


"""HERE IS reverse_attention_net"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class feature_branch(nn.Module):
    def __init__(self, input_channel, **kargs):
        super(feature_branch, self).__init__()
        self.net = Unet(input_channel=2, output_channel=8)

    def forward(self, x):
        return self.net(x)


class forward_branch(nn.Module):
    def __init__(self, input_channel, image_size=112, output_channel=64, **kargs):
        super(forward_branch, self).__init__()
        self.net = Unet(input_channel=input_channel, output_channel=output_channel)
        self.predict = Unet(input_channel=output_channel, output_channel=1)

    def forward(self, x):
        x = self.net(x)
        outputs = self.predict(x)
        return outputs, x


class reverse_branch(nn.Module):
    def __init__(self, input_channel, image_size=112, output_channel=64, **kargs):
        super(reverse_branch, self).__init__()
        self.image_size = image_size
        self.net = Unet(input_channel=input_channel, output_channel=output_channel)
        self.predict = Unet(input_channel=output_channel, output_channel=1)

    def forward(self, x):
        x = self.net(x)
        outputs = self.predict(x)
        return torch.ones_like(outputs) - outputs, x


class attention_branch(nn.Module):
    def __init__(self, input_channel, image_size=112, output_channel=64, **kargs):
        super(attention_branch, self).__init__()
        self.image_size = image_size
        self.attention = Unet(input_channel=output_channel, output_channel=output_channel)
        self.predict = Unet(input_channel=output_channel, output_channel=1)

    def forward(self, orginal_map, reverse_map):
        x = self.attention(torch.ones_like(orginal_map) - orginal_map)
        outputs = self.predict(x * reverse_map - orginal_map)
        return outputs


class reverse_attention_net(nn.Module):
    def __init__(self, image_size=224, pooling=True, input_channel=2, output_channel=64, bilinear=True, **kargs):
        super(reverse_attention_net, self).__init__()
        self.feature_net = feature_branch(input_channel=2)
        self.forward_net = forward_branch(input_channel=8, output_channel=output_channel)
        self.reverse_net = reverse_branch(input_channel=8, output_channel=output_channel)
        self.attention_net = attention_branch(input_channel=8, output_channel=output_channel)

    def forward(self, x):
        x = self.feature_net(x)
        orginal_predict, orginal_map = self.forward_net(x)
        reverse_predict, reverse_map = self.reverse_net(x)
        combine_predict = self.attention_net(orginal_map, reverse_map)
        return orginal_predict, reverse_predict, combine_predict

    def name(self):
        return "reverse_attention_net"


"""HERE IS TESTING"""

if __name__ == "__main__":
    # model = reverse_attention_net()
    # model.load_state_dict(torch.load('./model_RA'))
    # torch.save(model, './model_RA.pth')
    model = torch.load('./model_RA.pth')
