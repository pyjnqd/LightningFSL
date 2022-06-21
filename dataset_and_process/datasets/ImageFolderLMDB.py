import os
import os.path as osp
from PIL import Image
import six
import lmdb
import pickle
import numpy as np

import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)

"""
 继承Dataset，只是为了后续构建dataloader时适配
"""
class ImageFolderLMDB(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.db_path = root
        self.env = lmdb.open(self.db_path,
                             subdir=osp.isdir(self.db_path),
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        self.targets = []
        # self.samples = []
        with self.env.begin(write = False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))
            for idx in range(self.length):
                unpacked = loads_data(txn.get(self.keys[idx]))
                self.targets.append(unpacked[1])
                # self.samples.append(unpacked[0])
        self.transform = transform
        self.target_transform = target_transform
        # self.imgs = self.samples

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        # im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'




