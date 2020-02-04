import torch

import torch_points as tp

from src.core.base_conv.dense import BaseDenseConvolutionFlat
from src.core.neighbourfinder import DenseRadiusNeighbourFinder, DenseKNNNeighbourFinder


class DenseRandlanetKernel(torch.nn.Module):
    def __init__(self, rel_point_pos_nn=None, attention_nn=None, global_nn=None, *args, **kwargs):
        super().__init__()


class DenseRandlanetConv(BaseDenseConvolutionFlat):
    def __init__(self, k=None, radius=None, *args, **kwargs):
        super().__init__(DenseRadiusNeighbourFinder(radius, k) if radius else DenseKNNNeighbourFinder(k))

        self.kernel = DenseRandlanetKernel(**kwargs)

    def _prepare_features(self, x, pos, neigh_idx):
        pos_trans = pos.transpose(1, 2).contiguous()  # [B, 3, N]
        grouped_pos = tp.grouping_operation(pos_trans, neigh_idx)  # [B, 3, N, k]
        centroids = pos_trans.unsqueeze(-1)
        grouped_pos - centroids

        grouped_features = tp.grouping_operation(x, neigh_idx)  # []

    def conv(self, x, pos, neigh_idx):
        """
        Arguments:
            x -- Previous features [B, N, C]
            pos -- positions [B, N, 3]
            neigh_idx -- Indexes to group [B, N, k]
        Returns:
            new_x -- New features
        """
