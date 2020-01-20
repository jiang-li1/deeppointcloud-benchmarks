
from abc import ABC, abstractmethod
from typing import Optional, List
import math

import open3d as o3d
import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset

from datasets.base_dataset import BaseDataset
# from utils.pointcloud_utils import build_kdtree

class BasePointCloud(ABC):

    @property
    @abstractmethod
    def pos(self) -> torch.tensor:
        pass

    @property
    @abstractmethod
    def features(self) -> torch.tensor:
        pass

class BasePointCloudPatchDataset(torch.utils.data.Dataset, BasePointCloud, ABC):
    '''ABC for classes which generate patches from a single pointcloud

    PointCloudPatchDatasets should be backed by a torch_geometric.data.Data object 
    with non-None pos, this is the original pointcloud which will be sampled 
    into patches. 
    '''

    def __init__(self, data : Data):
        self._data = data

        assert data.pos is not None

    @property
    def data(self) -> Data:
        return self._data

    @property
    def pos(self) -> torch.tensor:
        return self.data.pos

    @property
    def features(self) -> torch.tensor:
        return self.data.x

    def get_bounding_box(self):
        minPoint = self.pos.min(dim=0)
        maxPoint = self.pos.max(dim=0)

        return minPoint.values, maxPoint.values

class BaseMultiCloudPatchDataset(ABC, Dataset):
    '''Class representing datasets over multiple patchable pointclouds. 

    This class basically forwards methods to the underlying list of patch datasets
    '''

    def __init__(self, patchDatasets: List[BasePointCloudPatchDataset]):
        self._patchDataset = patchDatasets

    @property
    def patch_datasets(self) -> List[BasePointCloudPatchDataset]:
        return self._patchDataset

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

    def __init__(self, data: Data, blockX, blockY, contextDist):
        super().__init__(data)

        self.blockXDist = blockX
        self.blockYDist = blockY
        self.contextDist = contextDist
        self.strideXDist = blockX - contextDist
        self.strideYDist = blockY - contextDist

        self.minPoint, self.maxPoint = self.get_bounding_box()

        cloudSizeX, cloudSizeY, _ = self.maxPoint - self.minPoint

        #number of blocks in the x dimension (grid columns)
        self.numBlocksX = math.ceil(cloudSizeX / self.strideXDist) 

        #number of blocks in the y dimension (grid rows) 
        self.numBlocksY = math.ceil(cloudSizeY / self.strideYDist) 

    def __len__(self):
        return self.numBlocksX * self.numBlocksY

    def _get_block_index_arr(self, idx):
        # xyMin = self.minPoint
        # xyMax = self.minPoint + torch.tensor([self.strideX, self.strideY, 0]).to(self.minPoint.dtype)
        # index = self.get_box_index(xyMin, xyMax)

        # return index

        index = self._get_box_index_arr(*self._get_bounds_for_idx(idx))

        return index

    def _get_inner_block_index_arr(self, idx):

        return self._get_box_index_arr(*self._get_inner_bounds_for_idx(idx))

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

        xyMin = (blockMinX, blockMinY)
        xyMax = (blockMaxX, blockMaxY)

        return xyMin, xyMax

    def _get_inner_bounds_for_idx(self, idx):
        xyMin, xyMax = self._get_bounds_for_idx(idx)
        return (
            (xyMin[0] + self.contextDist, xyMin[1] + self.contextDist), 
            (xyMax[0] - self.contextDist, xyMax[1] - self.contextDist)
        )

        

    def _get_box_index_arr(self, xyMin, xyMax):
        
        c1 = self.pos[:, 0] >= xyMin[0]
        c2 = self.pos[:, 0] <= xyMax[0]

        c3 = self.pos[:, 1] >= xyMin[1]
        c4 = self.pos[:, 1] <= xyMax[1]

        mask = c1 & c2 & c3 & c4

        return torch.arange(self.pos.shape[0])[mask]



    



    

        
    

    



