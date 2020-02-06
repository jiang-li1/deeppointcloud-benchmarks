
import math

import torch
import torch_geometric

from src.data.patch.base_patchable_pointcloud import BasePatchablePointCloud
from src.data.falible_dataset import InvalidIndexError

class Grid2DPatchableCloud(BasePatchablePointCloud):

    def __init__(self, data: torch_geometric.data.Data, blockX, blockY, contextDist, eval_mode=False, min_patch_size=2):
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
            raise InvalidIndexError("Patch at index: {} has {} (< min_patch_size) points".format(index, pos.shape[0]))

        xyMid = self._get_block_mid_for_idx(index)

        pos[:,0] -= xyMid[0]
        pos[:,1] -= xyMid[1]
        pos[:,2] -= pos.min(dim=0).values[2]

        d = torch_geometric.data.Data(
            pos = pos.contiguous(), 
            x = self.features[block_idx].to(torch.float).contiguous(),
            y = self.data.y[block_idx].to(torch.long).contiguous(),
            inner_idx = inner_idx.contiguous(),
        )

        if self.eval_mode:
            d.global_index = block_idx

        if hasattr(self.data, 'name'):
            d.name = self.data.name + ':{}'.format(index)

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