import timm
import torch
import torch.nn.functional as F


def timm_model():
    model = timm.create_model('vit_base_patch32_224', num_classes=0)
    model.outdim = 768
    return model
# x     = torch.randn(50, 3, 224, 224)
# print(model(x).shape)
# x = torch.unsqueeze(torch.unsqueeze(x, -1), -1)
# print(model(x).shape)
#
# x = F.adaptive_avg_pool2d(x, 1).squeeze_(-1).squeeze_(-1)
# print(model(x).shape)

def create_model():
    return timm_model()