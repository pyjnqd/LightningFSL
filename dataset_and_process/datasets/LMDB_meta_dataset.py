import os.path as osp
from PIL import Image
import six
import lmdb
import pickle
import numpy as np
from torchvision import transforms
import os
import torch.utils.data as data


def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)
def before_init(record_path):

    env = lmdb.open(record_path,
                    subdir=osp.isdir(record_path),
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)

    with env.begin(write=False) as txn:
        length = loads_data(txn.get(b'__len__'))
        keys = loads_data(txn.get(b'__keys__'))
    env.close()
    return length, keys

class LMDBClassDataset(data.Dataset):

    def __init__(self, record_path, transform=None):

        # self.env = lmdb.open(record_path,
        #                      subdir=osp.isdir(record_path),
        #                      readonly=True,
        #                      lock=False,
        #                      readahead=False,
        #                      meminit=False)
        #
        # with self.env.begin(write = False) as txn:
        #     self.length = loads_data(txn.get(b'__len__'))
        #     self.keys = loads_data(txn.get(b'__keys__'))

        # self.label = loads_data(txn.get(self.keys[0]))[1]

        self.record_path = record_path
        self.transform = transform
        self.length, self.keys = before_init(self.record_path)

    def open_lmdb(self):
        self.env = lmdb.open(self.record_path,
                             subdir=osp.isdir(self.record_path),
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)

        self.txn = self.env.begin(write = False, buffers=True)

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        # env = self.env
        # with env.begin(write=False) as txn:
        #     byteflow = txn.get(self.keys[index])

        byteflow = self.txn.get(self.keys[index])
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

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

class LMDBDataset(data.ConcatDataset):
    def __init__(self, split, dataset_dir, dataset_spec,  file_pattern = ".records"):
        dataset = []
        class_set = list(dataset_spec.get_classes(split))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if split == "TRAIN":
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
        else:
            transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])
        for record_file in os.listdir(dataset_dir):
            if ".records" in record_file:
                class_id, _ = record_file.split(".", 1)
                # print(class_id)
                class_id = int(class_id)
                if class_id in class_set and file_pattern in record_file:
                    class_dataset = LMDBClassDataset(osp.join(dataset_dir,record_file), transform)
                    dataset.append(class_dataset)
        super().__init__(dataset)

class MultipleLMDBDataset(data.ConcatDataset):
    def __init__(self, record_dir, file_pattern = ".records"):
        datasets = []
        for dataset_dir in os.listdir(record_dir):
            dataset = LMDBDataset(osp.join(record_dir, dataset_dir))
            datasets.append(dataset)
        self.dataset_num = len(datasets)
        super().__init__(datasets)

