from torchvision import transforms
import numpy as np
from .miniImageNet import miniImageNet
from PIL import ImageFilter
import random


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class JigCluTransform:
    def __init__(self, transform, cross = 0.0):
        self.transform = transform
        self.c = cross

    def __call__(self, x):
        h,w = x.size
        ch = self.c * h
        cw = self.c * w
        return [self.transform(x.crop((0,           0,          h//2+ch,    w//2+cw))),
                self.transform(x.crop((0,           w//2-cw,    h//2+ch,    w))),
                self.transform(x.crop((h//2-ch,     0,          h,          w//2+cw))),
                self.transform(x.crop((h//2-ch,     w//2-cw,    h,          w)))]

  
class miniImageNetMixedwithJigsaw(miniImageNet):
    r"""Dataset for contrastive learning mixed with normal training or few-shot learning.
    """
    def __init__(self, root: str, mode: str='train', image_sz = 84, patch_sz = 42) -> None:
        super().__init__(root, mode)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_sz),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                    np.array([0.2726, 0.2634, 0.2794]))])
        self.transform_jigsaw = transforms.Compose([
            transforms.RandomResizedCrop(image_sz, scale=(0.2, 1.0)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                np.array([0.2726, 0.2634, 0.2794]))
        ])
        self.transform_jigsaw = JigCluTransform(self.transform_jigsaw, 0.3) # 超参数0.3
        

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample_normal = self.transform(sample)
        sample_jigsaw = self.transform_jigsaw(sample)
        return sample_normal, target, sample_jigsaw



def return_class():
    return miniImageNetMixedwithJigsaw

