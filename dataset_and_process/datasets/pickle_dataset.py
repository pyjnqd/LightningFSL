import pickle
from torch.utils.data import Dataset
import torch
def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

class PickleDataset(Dataset):
    def __init__(self, root, mode = None):
        self.data, self.label = load_pickle(root)
        # print(self.data.shape)
        # print(type(self.label))
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), self.label[index]
    

def return_class():
    return PickleDataset