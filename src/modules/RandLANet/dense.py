import torch

import torch_points as tp
import etw_pytorch_utils as pt_utils

from src.core.base_conv.dense import BaseDenseConvolutionFlat
from src.core.neighbourfinder import DenseRadiusNeighbourFinder, DenseKNNNeighbourFinder


class DenseRandlanetKernel(torch.nn.Module):
    def __init__(self, rel_point_pos_nn=None, attention_nn=None, global_nn=None, *args, **kwargs):
        super().__init__()

        self.rel_point_pos_nn = pt_utils.SharedMLP(rel_point_pos_nn)

    def forward(self, rel_point_pos, grouped_features):
        """
            Arguments:
                rel_point_pos -- [B, 3+3+3+1, N, k]
                grouped_features -- [B, C, N, k]
        """

        self.rel_point_pos_nn(rel_point_pos)


class DenseRandlanetConv(BaseDenseConvolutionFlat):
    def __init__(self, k=None, radius=None, *args, **kwargs):
        super().__init__(DenseRadiusNeighbourFinder(radius, k) if radius else DenseKNNNeighbourFinder(k))

        self.kernel = DenseRandlanetKernel(**kwargs)

    def _prepare_features(self, x, pos, neigh_idx):
        pos_trans = pos.transpose(1, 2).contiguous()  # [B, 3, N]
        grouped_pos = tp.grouping_operation(pos_trans, neigh_idx)  # [B, 3, N, k]
        centroids = pos_trans.unsqueeze(-1)
        grouped_pos_relative = grouped_pos - centroids  # [B, 3, N, k]

        dists = torch.norm(grouped_pos_relative, p=2, dim=1).unsqueeze(1)

        rel_point_pos = torch.cat(
            [centroids, grouped_pos, grouped_pos_relative, dists], dim=1
        )  # [B, 3 + 3 + 3 + 1, N, k]

        grouped_features = tp.grouping_operation(x, neigh_idx)  # [B, C, N, k]

        return rel_point_pos, grouped_features

    def conv(self, x, pos, neigh_idx):
        """
        Arguments:
            x -- Previous features [B, N, C]
            pos -- positions [B, N, 3]
            neigh_idx -- Indexes to group [B, N, k]
        Returns:
            new_x -- New features
        """

        rel_point_pos, grouped_features = self._prepare_features(x, pos, neigh_idx)

        # rel_point_pos_enc = self.
