
from abc import ABC, abstractmethod
from typing import Optional, List
import math
import hashlib

import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
from overrides import overrides

from src.datasets.base_dataset import BaseDataset
# from utils.pointcloud_utils import build_kdtree

class BadDataException(Exception):

    def __init__(self, message):
        super().__init__(message)

class PointCloud():

    def __init__(self, pos, features=None):
        self._pos = pos
        self._features = features if features is not None else torch.tensor()
        self._minPoint = None
        self._maxPoint = None

        assert self._pos is not None

    @property
    def pos(self) -> torch.tensor:
        return self._pos

    @property
    def features(self) -> torch.tensor:
        return self._features

    @property
    def minPoint(self) -> torch.tensor:
        if self._minPoint is None:
            self.get_bounding_box()
        return self._minPoint

    @property
    def maxPoint(self) -> torch.tensor:
        if self._maxPoint is None:
            self.get_bounding_box()
        return self._maxPoint

    def get_bounding_box(self):
        minPoint = self.pos.min(dim=0)
        maxPoint = self.pos.max(dim=0)

        self._minPoint = minPoint.values
        self._maxPoint = maxPoint.values

        return self._minPoint, self._maxPoint

    def __len__(self):
        return self.pos.shape[0]

class ClassifiedPointCloud(PointCloud):

    def __init__(self, pos, classes, features=None):
        super().__init__(pos, features)

        self._classes = classes

    @classmethod
    def from_data(cls, data: Data):
        return cls(data.pos, data.y, data.x)

    @property
    def classes(self) -> torch.tensor:
        return self._classes


class BasePointCloudPatchDataset(torch.utils.data.Dataset, PointCloud, ABC):
    '''ABC for classes which generate patches from a single pointcloud

    PointCloudPatchDatasets should be backed by a torch_geometric.data.Data object 
    with non-None pos, this is the original pointcloud which will be sampled 
    into patches. 
    '''

    def __init__(self, data : Data):
        super().__init__(data.pos, data.x)
        self._data = data

    @property
    def data(self) -> Data:
        return self._data

    @property
    def num_features(self):
        return self.data.x.shape[1]


class BaseMultiCloudPatchDataset(ABC, torch.utils.data.Dataset):
    '''Class representing datasets over multiple patchable pointclouds. 

    This class basically forwards methods to the underlying list of patch datasets
    '''

    def __init__(self, patchDatasets: List[BasePointCloudPatchDataset]):
        self._patchDataset = patchDatasets

    @property
    def patch_datasets(self) -> List[BasePointCloudPatchDataset]:
        return self._patchDataset

    @property
    def num_features(self):
        return self.patch_datasets[0].num_features

    def __len__(self):
        return sum(len(pd) for pd in self.patch_datasets)

    def __getitem__(self, idx):
        
        i = 0

        for pds in self.patch_datasets:
            if idx < i + len(pds):
                return pds[idx - i]
            i += len(pds)
    

class BasePointCloudDataset(ABC):

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

class BasePatchDataset(ABC):

    def __init__(self):
        self._pointcloud_dataset = None

    @property
    @abstractmethod
    def pointcloud_dataset(self):
        return self._pointcloud_dataset

class BaseBallPointCloud(BasePointCloudPatchDataset, ABC):

    def __init__(self, pos):
        super().__init__(pos)

        self.kdtree = build_kdtree(self)

    def radius_query(self, point: torch.tensor, radius):
        k, indices, dist2 = self.kdtree.search_radius_vector_3d(point, radius)
        return k, indices, dist2

    def knn_query(self, point: torch.tensor, k):
        k, indices, dist2 = self.kdtree.search_knn_vector_3d(point, k)
        return k, indices, dist2

class BasePatchPointBallDataset(BasePatchDataset, ABC):
    '''
        Base class for patch datasets which return balls of points centered on
        points in the point clouds
    '''

    def __init__(self, pointcloud_dataset):
        super().__init__()

        self._pointcloud_dataset = pointcloud_dataset

    def __len__(self):
        return sum(len(cloud) for cloud in self.pointcloud_dataset)

    # def __getitem__(self, idx):

    #     i = 0

    #     for cloud in self.pointcloud_dataset:
    #         cloud : BasePointCloudDataset = cloud
    #         if idx < i + len(cloud):
    #             return cloud.

# class Grid2DPatchDataset(BasePatchDataset):

#     def __init__(self, backing_dataset: Dataset):
#         super().__init__(backing_dataset)

class Grid2DPatchDataset(BasePointCloudPatchDataset):

    def __init__(self, data: Data, blockX, blockY, contextDist, eval_mode=False, min_patch_size=2):
        super().__init__(data)

        self.eval_mode = eval_mode
        self.min_patch_size = min_patch_size

        self.blockXDist = blockX
        self.blockYDist = blockY
        self.contextDist = contextDist
        self.strideXDist = blockX - 2*contextDist
        self.strideYDist = blockY - 2*contextDist

        cloudSizeX, cloudSizeY, _ = self.maxPoint - self.minPoint

        #number of blocks in the x dimension (grid columns)
        self.numBlocksX = math.ceil(cloudSizeX / self.strideXDist) 

        #number of blocks in the y dimension (grid rows) 
        self.numBlocksY = math.ceil(cloudSizeY / self.strideYDist) 

    def __len__(self):
        return self.numBlocksX * self.numBlocksY

    def __getitem__(self, index):
        # print('Accessing {}'.format(index))
        block_idx = self._get_block_index_arr(index)
        inner_idx = self._get_inner_block_index_into_block(index)

        pos: torch.tensor = self.pos[block_idx].to(torch.float)

        if pos.shape[0] < self.min_patch_size:
            raise BadDataException("Patch at index: {} has {} (< min_patch_size) points".format(index, pos.shape[0]))

        xyMid = self._get_block_mid_for_idx(index)

        pos[:,0] -= xyMid[0]
        pos[:,1] -= xyMid[1]
        pos[:,2] -= pos.min(dim=0).values[2]

        d = Data(
            pos = pos.contiguous(), 
            x = self.features[block_idx].to(torch.float).contiguous(),
            y = self.data.y[block_idx].to(torch.long).contiguous(),
            inner_idx = inner_idx.contiguous(),
        )

        if self.eval_mode:
            d.global_index = block_idx

        return d

    def _get_block_index_arr(self, idx):
        return self._get_box_index_arr(*self._get_bounds_for_idx(idx))

    def _get_inner_block_index_arr(self, idx):
        return self._get_box_index_arr(*self._get_inner_bounds_for_idx(idx))

    def _get_inner_block_index_into_block(self, idx):
        block_index = self._get_block_index_arr(idx)
        return self._get_box_index_arr(
            *self._get_inner_bounds_for_idx(idx),
            pts=self.pos[block_index]
        )

    def _get_block_mid_for_idx(self, idx):

        xyMin, xyMax = self._get_bounds_for_idx(idx)

        xyMid = (
            xyMin[0] + (
                (xyMax[0] - xyMin[0]) / 2
            ), 
            xyMin[1] + (
                (xyMax[1] - xyMin[1]) / 2
            )
        )

        return xyMid

    def _get_bounds_for_idx(self, idx):
        yIndex, xIndex = divmod(idx, self.numBlocksX)

        blockMinY = self.minPoint[1] + yIndex * self.strideYDist
        blockMinX = self.minPoint[0] + xIndex * self.strideXDist

        blockMaxY = torch.min(
            blockMinY + self.blockYDist,
            self.maxPoint[1]
        )
        blockMaxX = torch.min(
            blockMinX + self.blockXDist,
            self.maxPoint[0]
        )

        blockMinY = blockMaxY - self.blockYDist
        blockMinX = blockMaxX - self.blockXDist

        xyMin = (blockMinX, blockMinY)
        xyMax = (blockMaxX, blockMaxY)

        return xyMin, xyMax

    def _get_inner_bounds_for_idx(self, idx):
        xyMin, xyMax = self._get_bounds_for_idx(idx)
        return (
            (xyMin[0] + self.contextDist, xyMin[1] + self.contextDist), 
            (xyMax[0] - self.contextDist, xyMax[1] - self.contextDist)
        )

    def _get_box_index_arr(self, xyMin, xyMax, pts=None):

        if pts is None:
            pts = self.pos
        
        c1 = pts[:, 0] >= xyMin[0]
        c2 = pts[:, 0] <= xyMax[0]

        c3 = pts[:, 1] >= xyMin[1]
        c4 = pts[:, 1] <= xyMax[1]

        mask = c1 & c2 & c3 & c4

        return torch.arange(pts.shape[0])[mask]


class FailSafeIterableDataset(torch.utils.data.IterableDataset):
    '''Creates an IterableDataset around an ordinary map-style dataset
    where some indicies point to bad data. For example a patch dataset
    where some patches are empty. 

    The underlying dataset should throw BadDataException to indicate
    that the index is bad. FailSafeIterableDataset will try retry 
    up to max_retries times to fetch an item from the dataset. 

    '''

    def __init__(self, dataset: torch.utils.data.Dataset, sampler: torch.utils.data.Sampler):
        self._sampler = iter(sampler)
        self._dataset = dataset
        self._max_retries = 10

    def __iter__(self):
        return self

    def __next__(self):

        for _ in range(self._max_retries):

            idx = next(self._sampler)

            try:
                return self._dataset[idx]
            except BadDataException as e:
                print('Skipping bad data sample', print(str(e)))
                continue
        
        raise BadDataException("Dataset returned BadDataException more times than _max_retries")

    #forward all attribute calls to the underlying dataset
    #(e.g. num_features)
    def __getattr__(self, name):
        return getattr(self._dataset, name)

class UniqueRandomSampler(torch.utils.data.RandomSampler):
    '''
    Random sampler which producess unique indexes even when
    duplicated across torch dataloader worker threads. 

    torch.utils.data.RandomSampler will produce the same sequence
    of indexes in each worker thread.

    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._offset = None

    def _get_offset(self):

        if self._offset is not None:
            return self._offset

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            self._offset = 0
        else:      
            self._offset = int(
                hashlib.md5(
                    str(worker_info.id).encode()
                ).hexdigest(),
                16
            )     

        return self._offset

    @overrides
    def __iter__(self):
        return (
            (idx + self._get_offset()) % len(self.data_source)
            for idx in super().__iter__()
        )





        



    



    

        
    

    



