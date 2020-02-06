
from abc import ABC

import torch
import torch_geometric

from src.data.pointcloud import PointCloud

class BasePatchablePointCloud(torch.utils.data.Dataset, PointCloud, ABC):
    '''ABC for classes which generate patches from a single pointcloud.

    PointCloudPatchDatasets should be backed by a torch_geometric.data.Data object 
    with non-None pos, this is the original pointcloud which will be sampled 
    into patches. 
    '''

    def __init__(self, data : torch_geometric.data.Data):
        super().__init__(data.pos, data.x)
        self._data = data

    @property
    def data(self) -> torch_geometric.data.Data:
        return self._data

    @property
    def num_features(self):
        return self.data.x.shape[1]

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()


class BaseIterablePatchablePointCloud(torch.utils.data.IterableDataset, PointCloud, ABC):
    '''ABC for classes which generate patches from a single pointcloud
    
    '''

    def __init__(self, data: torch_geometric.data.Data):
        raise NotImplementedError()


# class BaseLazyPointCloudPatchDataset(torch.utils.data.Dataset, PointCloud, ABC):
#     '''Acts like BasePointCloudPatchDataset, with data = dataset[idx], except that
#     data will only be loaded when required. This supports large datasets which 
#     cannot fit into memory. 

#     '''

#     def __init__(self, dataset: torch.utils.data.Dataset, idx):
#         self._dataset = dataset
#         self._idx = idx 
#         self._data = None

#     def load(self):
#         self._data = self._dataset[self._idx]

#     def unload(self):
#         self._data = None

#     @property
#     def pos(self) -> torch.tensor:
#         return self.data.pos

#     @property
#     def features(self) -> torch.tensor:
#         return self.data.features

#     @property
#     def data(self) -> Data:
#         if self._data is None:
#             self.load()
#         return self._data

#     @property
#     def num_features(self):
#         return self.data.x.shape[1]

#     def __len__(self):
#         raise NotImplementedError()

#     def __getitem__(self, index):
#         raise NotImplementedError()
