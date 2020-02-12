import torch
import torch.nn.functional as F
import torch_points as tp
import etw_pytorch_utils as pt_utils

from src.core.base_conv.dense import BaseDenseConvolutionFlat
from src.core.neighbourfinder import DenseRadiusNeighbourFinder, DenseKNNNeighbourFinder, BaseNeighbourFinder
from src.core.sampling import BaseSampler
from src.core.base_conv.message_passing import BaseModule

class RandlaKernelDense(BaseModule):
    """
        Implements both the Local Spatial Encoding and Attentive Pooling blocks from
        RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds
        https://arxiv.org/pdf/1911.11236

    """

    def __init__(self, rel_point_pos_nn=None, attention_nn=None, global_nn=None, *args, **kwargs):
        super().__init__()

        make_mlp = pt_utils.SharedMLP

        self.rel_point_pos_nn = make_mlp(rel_point_pos_nn)
        self.attention_nn = make_mlp(attention_nn)
        self.global_nn = make_mlp(global_nn)

    # def forward(self, x, pos, edge_index):
    #     x = self.propagate(edge_index, x=x, pos=pos)
    #     return x

    def forward(self, x, pos, neigh_idx):
        """
        Arguments:
            x -- Previous features [B, N, C]
            pos -- positions [B, N, 3]
            neigh_idx -- Indexes to group [B, N, k]
        Returns:
            new_x -- New features
        """

        rel_point_pos, grouped_features = self._prepare_features(x, pos, neigh_idx)
        # rel_point_pos: [B, 10, N, k]
        # grouped_features: [B, C, N, k]

        rel_point_pos_enc = self.rel_point_pos_nn(rel_point_pos) # [B, C, N, k]

        fij_hat = torch.cat([grouped_features, rel_point_pos_enc], dim=1)

        g_fij = self.attention_nn(fij_hat)
        s_ij = F.softmax(g_fij, -1)

        msg = s_ij * fij_hat

        new_features = F.avg_pool2d(msg, kernel_size=[1, msg.size(3)])

        new_features = self.global_nn(new_features)

        return new_features


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

    # def message(self, x_j, pos_i, pos_j):

    #     if x_j is None:
    #         x_j = pos_j

    #     # compute relative position encoding
    #     vij = pos_i - pos_j

    #     dij = torch.norm(vij, dim=1).unsqueeze(1)

    #     relPointPos = torch.cat([pos_i, pos_j, vij, dij], dim=1)

    #     rij = self.rel_point_pos_nn(relPointPos)

    #     # concatenate position encoding with feature vector
    #     fij_hat = torch.cat([x_j, rij], dim=1)

    #     # attentative pooling
    #     g_fij = self.attention_nn(fij_hat)
    #     s_ij = F.softmax(g_fij, -1)

    #     msg = s_ij * fij_hat

    #     return msg

    # def update(self, aggr_out):
    #     return self.global_nn(aggr_out)

# class DenseRandlanetKernel(torch.nn.Module):
#     def __init__(self, rel_point_pos_nn=None, attention_nn=None, global_nn=None, *args, **kwargs):
#         super().__init__()

#         self.rel_point_pos_nn = pt_utils.SharedMLP(rel_point_pos_nn)

#     def forward(self, rel_point_pos, grouped_features):
#         """
#             Arguments:
#                 rel_point_pos -- [B, 3+3+3+1, N, k]
#                 grouped_features -- [B, C, N, k]
#         """

#         self.rel_point_pos_nn(rel_point_pos)


# class DenseRandlanetConv(BaseDenseConvolutionFlat):
#     def __init__(self, k=None, radius=None, *args, **kwargs):
#         super().__init__(DenseRadiusNeighbourFinder(radius, k) if radius else DenseKNNNeighbourFinder(k))

#         self.kernel = DenseRandlanetKernel(**kwargs)

#     def _prepare_features(self, x, pos, neigh_idx):
#         pos_trans = pos.transpose(1, 2).contiguous()  # [B, 3, N]
#         grouped_pos = tp.grouping_operation(pos_trans, neigh_idx)  # [B, 3, N, k]
#         centroids = pos_trans.unsqueeze(-1)
#         grouped_pos_relative = grouped_pos - centroids  # [B, 3, N, k]

#         dists = torch.norm(grouped_pos_relative, p=2, dim=1).unsqueeze(1)

#         rel_point_pos = torch.cat(
#             [centroids, grouped_pos, grouped_pos_relative, dists], dim=1
#         )  # [B, 3 + 3 + 3 + 1, N, k]

#         grouped_features = tp.grouping_operation(x, neigh_idx)  # [B, C, N, k]

#         return rel_point_pos, grouped_features

#     def conv(self, x, pos, neigh_idx):
#         """
#         Arguments:
#             x -- Previous features [B, N, C]
#             pos -- positions [B, N, 3]
#             neigh_idx -- Indexes to group [B, N, k]
#         Returns:
#             new_x -- New features
#         """

#         rel_point_pos, grouped_features = self._prepare_features(x, pos, neigh_idx)

#         # rel_point_pos_enc = self.

class RandlaBlockDense(BaseModule):

    def __init__(self, *, 
        sampler: BaseSampler = None,
        in_reshape_nn,
        neighbour_finder: BaseNeighbourFinder,
        rel_point_pos_nn_1,
        attention_nn_1,
        aggregation_nn_1,
        rel_point_pos_nn_2,
        attention_nn_2,
        aggregation_nn_2,
        skip_nn,
        **kwargs
    ):
        super().__init__()

        make_mlp = pt_utils.SharedMLP

        self.sampler = sampler
        self.in_reshape_nn = make_mlp(in_reshape_nn)
        self.neighbour_finder = neighbour_finder

        self.kernel_1 = RandlaKernelDense(
            rel_point_pos_nn=rel_point_pos_nn_1,
            attention_nn=attention_nn_1,
            global_nn=aggregation_nn_1
        )
        
        self.kernel_2 = RandlaKernelDense(
            rel_point_pos_nn=rel_point_pos_nn_2,
            attention_nn=attention_nn_2,
            global_nn=aggregation_nn_2
        )

        self.skip_nn = make_mlp(skip_nn)

    def forward(self, data):
        x, pos = data.x, data.pos

        if self.sampler is not None:
            idx = self.sampler(data.pos, batch)
            x = x[idx]
            pos = pos[idx]
            batch = batch[idx]

        shortcut = x

        x = self.in_reshape_nn(x) # (N, L_OUT//4)

        radius_idx =self.neighbour_finder(pos, pos)
        x = self.kernel_1(x, pos, radius_idx) # (N, L_OUT//2)
        x = self.kernel_2(x, pos, radius_idx) # (N, L_OUT)

        shortcut = self.skip_nn(shortcut) # (N, L_OUT)
        x = shortcut + x

        return Data(pos=pos, x=x)

    def extra_repr(self):
        return '\n'.join([
            '(sampler): {}'.format(repr(self.sampler)),
            '(neighbour_finder): {}'.format(repr(self.neighbour_finder)),
        ])
