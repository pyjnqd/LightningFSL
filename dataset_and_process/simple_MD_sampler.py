from torch.utils.data import Sampler
import torch.distributed as dist
from typing import List, Union, TypeVar, Iterator
from collections import defaultdict
import math
import torch
import numpy as np
T_co = TypeVar('T_co', covariant=True)

MAX_SPANNING_LEAVES_ELIGIBLE = 392

class MetaDatasetSampler(Sampler[T_co]):
    """
    The sampler that neglects topology of class structure, does not support multiple dataset training
    and does not support varying ways and shots during sampling.
    """
    def __init__(self, 
        dataset_spec,
        split,
        num_task,
        way,
        total_sample_per_class,
        total_batch_size,
        is_DDP,
        drop_last,
        ):
        self.dataset_spec = dataset_spec
        self.split = split
        self.num_task = num_task
        self.way = way
        self.total_sample_per_class = total_sample_per_class
        self.total_batch_size = total_batch_size
        self.is_DDP = is_DDP
        self.drop_last = drop_last
        self.per_gpu_batch_size = self.total_batch_size

        if self.is_DDP:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            self.num_replicas = dist.get_world_size()
            assert self.total_batch_size % self.num_replicas == 0
            self.per_gpu_batch_size = self.total_batch_size//self.num_replicas


        if self.drop_last and self.num_task % self.total_batch_size != 0:
            self.num_iteration = math.ceil(
                (self.num_task - self.total_batch_size) / self.total_batch_size  # type: ignore
            )
        else:
            self.num_iteration = math.ceil(self.num_task / self.total_batch_size)
        
        class_set = self.dataset_spec.get_classes(self.split)
        self.m_ind = []#the data index of each class
        count = 0
        for class_id in class_set:
            num_images = self.dataset_spec.get_total_images_per_class(class_id)
            if num_images >= self.way * self.total_sample_per_class:
                ind = np.arange(count, count+num_images)
                ind = torch.from_numpy(ind)
                count += num_images
                self.m_ind.append(ind)
        if len(self.m_ind)<self.way:
            raise ValueError(f'There are no classes eligible for participating in '
                                    'episodes for dataset {i}. Consider change self.way')
    
    def __len__(self) -> int:
        return self.num_iteration
    
    def __iter__(self) -> Iterator[T_co]:
        # print(self.num_iteration)
        for _ in range(self.num_iteration):
            tasks = []
            for _ in range(self.per_gpu_batch_size):
                task = []
                #random sample num_class indexs,e.g. 5
                classes = torch.randperm(len(self.m_ind))[:self.way]
                for c in classes:
                    #sample total_sample_per_class data index of this class
                    l = self.m_ind[c]#all data indexs of this class
                    pos = torch.randperm(len(l))[:self.total_sample_per_class] 
                    # print(pos)
                    task.append(l[pos])
                tasks.append(torch.stack(task).t().reshape(-1))
            tasks = torch.stack(tasks).reshape(-1)
            yield tasks







