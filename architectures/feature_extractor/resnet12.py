import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        return out


class ResNet12(nn.Module):
    """The standard popular ResNet12 Model used in Few-Shot Learning.
    """
    def __init__(self, channels):
        super().__init__()

        self.inplanes = 3

        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])

        self.outdim = channels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4


def create_model():
    return ResNet12([64, 160, 320, 640])

