"""
    swin_tiny_patch4_window7_224
"""

import timm
import torch


def timm_model():
    model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=0)
    model.outdim = 768
    return model
# x     = torch.randn(1, 3, 224, 224)
# print(timm_model()(x).shape)


def create_model():
    return timm_model()