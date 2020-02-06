
from typing import Optional, List, Callable
import math
import hashlib

import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
from overrides import overrides
import numpy as np

from src.data.base_dataset import BaseDataset
from src.data.patch.base_patchable_pointcloud import BasePatchablePointCloud
from src.data.falible_dataset import FalibleDatasetWrapper
from src.data.sampler import unique_random_index, epoch_unique_random_seed


class PatchDataset(torch.utils.data.Dataset):
    '''Class representing datasets over multiple patchable pointclouds. 

    This class basically forwards methods to the underlying list of patch datasets

    A dataset will usually consist of multiple pointclouds, each of which must be sampled
    as patches. Each pointcloud is represented by a BasePatchablePointCloud. This class provides 
    an interface to a list of BasePatchablePointClouds
    '''

    def __init__(self, patchable_clouds: List[BasePatchablePointCloud]):
        self._patchable_clouds = patchable_clouds

    @property
    def patchable_clouds(self) -> List[BasePatchablePointCloud]:
        return self._patchable_clouds

    def __len__(self):
        return sum(len(pd) for pd in self.patchable_clouds)

    def __getitem__(self, idx):
        
        i = 0

        for pds in self.patchable_clouds:
            if idx < i + len(pds):
                return pds[idx - i]
            i += len(pds)

    #forward all attribute calls to the underlying datasets
    #(e.g. num_features)
    def __getattr__(self, name):
        return getattr(self.patchable_clouds[0], name)

class PartialPatchDataset(torch.utils.data.Dataset):
    '''like BaseMultiCloudPatchDatasets, but for datasets that are too large to fit in memory''' 

    def __init__(self, 
        backing_dataset: torch.utils.data.Dataset, 
        make_patchable_cloud: Callable[[Data], BasePatchablePointCloud],
        samples_per_dataset = 10,
        datasets_per_worker = 2
    ):
        self._backing_dataset = backing_dataset
        self._make_patchable_cloud = make_patchable_cloud
        self._samples_per_dataset = samples_per_dataset
        self._num_loaded_datasets = datasets_per_worker

        self._patch_dataset = None
        self._index_sequence = []
        # self._load()

        self.num_classes = 5
        self.num_features = 3

    def __len__(self):
        return self._samples_per_dataset * self._num_loaded_datasets

    @overrides
    def __getitem__(self, idx):
        if self._patch_dataset is None:
            self._load()
        return self._patch_dataset[self._index_sequence[idx]]


    # def __next__(self):
    #     if self._num_samples_taken > self._samples_per_dataset * self._num_loaded_datasets:
    #         self.cycle()

    #     idx = unique_random_index(len(self._patch_dataset))
    #     self._num_samples_taken += 1
    #     return self._patch_dataset[idx]

    #forward all attribute calls to the underlying datasets
    #(e.g. num_features)
    def __getattr__(self, name):
        return getattr(self._patch_dataset, name)

    def _load(self):
        s = epoch_unique_random_seed() % 2**31
        np.random.seed(s)
        patchableCloudIndexes = np.random.choice(
                len(self._backing_dataset),
                size=self._num_loaded_datasets,
                replace=False,
        )

        self._patch_dataset = PatchDataset([
            self._make_patchable_cloud(self._backing_dataset[idx]) 
            for idx in patchableCloudIndexes
        ])

        self._index_sequence = np.random.choice(
            len(self._patch_dataset),
            self._samples_per_dataset * self._num_loaded_datasets,
            replace=False
        )

    def reset(self):
        # self._load()
        pass







        



    



    

        
    

    



