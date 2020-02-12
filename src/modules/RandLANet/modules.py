import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from src.core.spatial_ops import *
from src.core.base_conv.message_passing import *

from src.utils.model_examination import LayerInfo


class RandlaKernel(MessagePassing):
    """
        Implements both the Local Spatial Encoding and Attentive Pooling blocks from
        RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds
        https://arxiv.org/pdf/1911.11236

    """

    def __init__(self, rel_point_pos_nn=None, attention_nn=None, global_nn=None, *args, **kwargs):
        MessagePassing.__init__(self, aggr="add")

        self.rel_point_pos_nn = MLP(rel_point_pos_nn)
        self.attention_nn = MLP(attention_nn)
        self.global_nn = MLP(global_nn)

    def forward(self, x, pos, edge_index):
        x = self.propagate(edge_index, x=x, pos=pos)
        return x

    def message(self, x_j, pos_i, pos_j):

        if x_j is None:
            x_j = pos_j

        # compute relative position encoding
        vij = pos_i - pos_j

        dij = torch.norm(vij, dim=1).unsqueeze(1)

        relPointPos = torch.cat([pos_i, pos_j, vij, dij], dim=1)

        rij = self.rel_point_pos_nn(relPointPos)

        # concatenate position encoding with feature vector
        fij_hat = torch.cat([x_j, rij], dim=1)

        # attentative pooling
        g_fij = self.attention_nn(fij_hat)
        s_ij = F.softmax(g_fij, -1)

        msg = s_ij * fij_hat

        return msg

    def update(self, aggr_out):
        return self.global_nn(aggr_out)


class RandlaConv(BaseConvolutionDown):
    def __init__(self, ratio=None, k=None, *args, **kwargs):
        super(RandlaConv, self).__init__(RandomSampler(ratio), KNNNeighbourFinder(k), *args, **kwargs)
        if kwargs.get("index") == 0 and kwargs.get("nb_feature") is not None:
            kwargs["point_pos_nn"][-1] = kwargs.get("nb_feature")
            kwargs["attention_nn"][0] = kwargs["attention_nn"][-1] = kwargs.get("nb_feature") * 2
            kwargs["down_conv_nn"][0] = kwargs.get("nb_feature") * 2
        self._conv = RandlaKernel(*args, global_nn=kwargs["down_conv_nn"], **kwargs)

    def conv(self, x, pos, edge_index, batch):
        return self._conv(x, pos, edge_index)


class DilatedResidualBlock(BaseResnetBlock):
    def __init__(
        self,
        indim,
        outdim,
        ratio1,
        ratio2,
        point_pos_nn1,
        point_pos_nn2,
        attention_nn1,
        attention_nn2,
        global_nn1,
        global_nn2,
        *args,
        **kwargs
    ):
        if kwargs.get("index") == 0 and kwargs.get("nb_feature") is not None:
            indim = kwargs.get("nb_feature")
        super(DilatedResidualBlock, self).__init__(indim, outdim, outdim)
        self.conv1 = RandlaConv(
            ratio1, 16, point_pos_nn=point_pos_nn1, attention_nn=attention_nn1, down_conv_nn=global_nn1, *args, **kwargs
        )
        kwargs["nb_feature"] = None
        self.conv2 = RandlaConv(
            ratio2, 16, point_pos_nn=point_pos_nn2, attention_nn=attention_nn2, down_conv_nn=global_nn2, *args, **kwargs
        )

    def convs(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        return data


class RandLANetRes(torch.nn.Module):
    def __init__(self, indim, outdim, ratio, point_pos_nn, attention_nn, down_conv_nn, *args, **kwargs):
        super(RandLANetRes, self).__init__()

        self._conv = DilatedResidualBlock(
            indim,
            outdim,
            ratio[0],
            ratio[1],
            point_pos_nn[0],
            point_pos_nn[1],
            attention_nn[0],
            attention_nn[1],
            down_conv_nn[0],
            down_conv_nn[1],
            *args,
            **kwargs
        )

    def forward(self, data):
        return self._conv.forward(data)

class RandlaBlock(BaseModule):

    def __init__(self, *, 
        sampler: MaskBaseSampler = None,
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

        make_mlp = MLP

        self.sampler = sampler
        self.in_reshape_nn = make_mlp(in_reshape_nn)
        self.neighbour_finder = neighbour_finder

        self.kernel_1 = RandlaKernel(
            rel_point_pos_nn=rel_point_pos_nn_1,
            attention_nn=attention_nn_1,
            global_nn=aggregation_nn_1
        )
        
        self.kernel_2 = RandlaKernel(
            rel_point_pos_nn=rel_point_pos_nn_2,
            attention_nn=attention_nn_2,
            global_nn=aggregation_nn_2
        )

        self.skip_nn = make_mlp(skip_nn)

        self.EXAMINE_MODE = True
        self.layer_info = None

    def forward(self, data):
        batch_obj = Batch()
        x, pos, batch = data.x, data.pos, data.batch

        if self.sampler is not None:
            idx = self.sampler(data.pos, batch)
            batch_obj.idx = idx
            x = x[idx]
            pos = pos[idx]
            batch = batch[idx]

        shortcut = x

        x = self.in_reshape_nn(x) # (N, L_OUT//4)

        row, col = self.neighbour_finder(pos, pos, batch, batch)
        edge_index = torch.stack([col, row], dim=0)
        x = self.kernel_1(x, pos, edge_index) # (N, L_OUT//2)
        x = self.kernel_2(x, pos, edge_index) # (N, L_OUT)

        shortcut = self.skip_nn(shortcut) # (N, L_OUT)
        x = shortcut + x

        if self.EXAMINE_MODE:
            # self.EXAMINE_DICT['pos'] = pos
            # self.EXAMINE_DICT['edge_index'] = edge_index
            # self.EXAMINE_DICT['samp_idx'] = idx if 'idx' in locals() else None
            self.layer_info = LayerInfo(
                pos,
                idx if 'idx' in locals() else None,
                edge_index,
            )

        batch_obj.x = x
        batch_obj.pos = pos
        batch_obj.batch = batch
        copy_from_to(data, batch_obj)
        return batch_obj

    def extra_repr(self):
        return '\n'.join([
            '(sampler): {}'.format(repr(self.sampler)),
            '(neighbour_finder): {}'.format(repr(self.neighbour_finder)),
        ])
        

class RandlaBaseModule(BaseModule):

    def __init__(self, *, nn, **kwargs):
        super().__init__()
        self.nn = MLP(nn)

    def forward(self, data):
        data.x = self.nn(data.x)
        return data

