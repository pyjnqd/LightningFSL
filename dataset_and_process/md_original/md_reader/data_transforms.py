import torch
from torchvision import transforms
import random
from PIL import ImageFilter
import numpy as np
from utils import device
import gin
from meta_dataset.data.config import DataAugmentation


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

"""
基于torch.transform实现的图像变换，根据不同算法设置不同变换
input: list of images(PIL Image)
output: torch.Tensor
"""

def jigsaw_transform(images: list):

    transform_patch = transforms.Compose([
        transforms.RandomResizedCrop(42, scale=(0.2, 1.0)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(np.array([0.5, 0.5, 0.5]),
                             np.array([0.5, 0.5, 0.5]))
    ])
    patches = []
    for img in images:
        h, w = img.size
        ch = 0.25 * h
        cw = 0.25 * w
        one_patches = [transform_patch(img.crop((0, 0, h // 2 + ch, w // 2 + cw))).to(device),
                       transform_patch(img.crop((0, w // 2 - cw, h // 2 + ch, w))).to(device),
                       transform_patch(img.crop((h // 2 - ch, 0, h, w // 2 + cw))).to(device),
                       transform_patch(img.crop((h // 2 - ch, w // 2 - cw, h, w))).to(device)]
        patches.append(one_patches)

    return patches


def tsa_episode_transform(
        images: list,
        mode: str
):
    assert mode in ['support', 'query']
    imgs = []
    if mode == 'support':
        for img in images:
            img = transforms.functional.to_tensor(img)
            img = transforms.functional.resize(img, (84, 84))
            img = transforms.functional.normalize(img, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
            # if support_data_augmentation.data_augmentation.enable_gaussian_noise:
            #     # TODO: implement aug in torch
            #     pass
            # if support_data_augmentation.data_augmentation.enable_jitter:
            #     # TODO: implement aug in torch
            #     pass
            imgs.append(img)
    else:
        for img in images:
            img = transforms.functional.to_tensor(img)
            img = transforms.functional.resize(img, (84, 84))
            img = transforms.functional.normalize(img, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
            # if query_data_augmentation.data_augmentation.enable_gaussian_noise:
            #     # TODO: implement aug in torch
            #     pass
            # if query_data_augmentation.data_augmentation.enable_jitter:
            #     # TODO: implement aug in torch
            #     pass
            imgs.append(img)

    return torch.stack(imgs).to(device)