import timm
import torch
import torch.nn.functional as F


def timm_model():
    model = timm.create_model('resnet50', num_classes=0)
    model.outdim = 2048
    return model
# x     = torch.randn(50, 3, 224, 224)
# print(timm_model()(x).shape)
# x = torch.unsqueeze(torch.unsqueeze(x, -1), -1)
# print(model(x).shape)


def create_model():
    return timm_model()