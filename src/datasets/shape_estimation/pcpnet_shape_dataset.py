import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.datasets import PCPNetDataset

from .base_patch_dataset import BasePointCloudDataset, BasePatchDataset, BasePointCloud
from .base_dataset import BaseDataset

class PCPNetShape(BasePointCloud):

    def __init__(self, data):
        super().__init__()

        self._pos = data.pos
        self._normals = data.x[:,:3]
        self._curv = data.x[:,3:5]
        self._pidx = data.test_idx

    @property
    def normals(self):
        return self._normals

    @property
    def curv(self):
        return self._curv

    @property
    def pidx(self):
        return self._pidx

class PCPNetCloudDataset(BasePointCloudDataset):

    def __init__(self, split, category):
        super().__init__()

        self._data_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PCPNetDataset')

        self._backing_dataset = PCPNetDataset(
            self._data_path,
            category,
            split=split
        )

    def __len__(self):
        return len(self._backing_dataset)

    def __getitem__(self, idx):
        return PCPNetShape(self._backing_dataset[idx])


class PCPNetPatchDataset(BasePatchDataset):

    def __init__(self, category, split):
        super().__init__()

        self._pointcloud_dataset = PCPNetCloudDataset(split, category)

class PCPNetShapeDataset(BaseDataset):

    def __init__(self, dataset_opt, training_opt):

        self.train_dataset = PCPNetPatchDataset(
            dataset_opt.train_category,
            split = 'train',
        )

        self.val_dataset = PCPNetPatchDataset(
            dataset_opt.train_category,
            split = 'val',
        )

        self.test_dataset = PCPNetPatchDataset(
            dataset_opt.test_category,
            split = 'test'
        )






