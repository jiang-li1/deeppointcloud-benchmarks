import torch

import torch.nn.functional as F
from torch_geometric.data import Data
import etw_pytorch_utils as pt_utils
import logging

from src.modules.pointnet2 import *
from src.core.base_conv.dense import DenseFPModule
from src.models.base_architectures import UnetBasedModel
from src.core.losses.losses import FocalLoss
from .base import Segmentation_MP

log = logging.getLogger(__name__)


class PointNet2_D(UnetBasedModel):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnetBasedModel
        UnetBasedModel.__init__(self, option, model_type, dataset, modules, superbatch_size=1)
        self._num_classes = dataset.num_classes
        self._weight_classes = dataset.weight_classes
        self._use_category = option.use_category
        if self._use_category:
            if not dataset.class_to_segments:
                raise ValueError(
                    "The dataset needs to specify a class_to_segments property when using category information for segmentation"
                )
            self._num_categories = len(dataset.class_to_segments.keys())
            log.info("Using category information for the predictions with %i categories", self._num_categories)
        else:
            self._num_categories = 0

        # Last MLP
        last_mlp_opt = option.mlp_cls

        self.FC_layer = pt_utils.Seq(last_mlp_opt.nn[0] + self._num_categories)
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.conv1d(last_mlp_opt.nn[i], bn=True)
        if last_mlp_opt.dropout:
            self.FC_layer.dropout(p=last_mlp_opt.dropout)

        self.FC_layer.conv1d(self._num_classes, activation=None)
        self.loss_names = ["loss_seg"]

        self.lossModule = FocalLoss(gamma=2)

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        Sets:
            self.input:
                x -- Features [B, C, N]
                pos -- Points [B, N, 3]
        """
        assert len(data.pos.shape) == 3
        self.input = Data(x=data.x.transpose(1, 2).contiguous(), pos=data.pos)
        self.labels = torch.flatten(data.y).long()  # [B * N]
        self.batch_idx = torch.arange(0, data.pos.shape[0]).view(-1, 1).repeat(1, data.pos.shape[1]).view(-1)
        if self._use_category:
            self.category = data.category

    def forward(self):
        r"""
            Forward pass of the network
            self.input:
                x -- Features [B, C, N]
                pos -- Points [B, N, 3]
        """
        data = self.model(self.input)
        last_feature = data.x
        if self._use_category:
            cat_one_hot = F.one_hot(self.category, self._num_categories).float().transpose(1, 2)
            last_feature = torch.cat((last_feature, cat_one_hot), dim=1)

        self.output = self.FC_layer(last_feature).transpose(1, 2).contiguous().view((-1, self._num_classes))

        self.output = F.softmax(self.output, dim=-1)
        
        if self._weight_classes is not None:
            self._weight_classes = self._weight_classes.to(self.output.device)
        # self.loss_seg = F.cross_entropy(self.output, self.labels, weight=self._weight_classes)
        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        if self._weight_classes is not None:
            self._weight_classes = self._weight_classes.to(self.output.device)
        # self.loss_seg = F.cross_entropy(self.output, self.labels, weight=self._weight_classes)

        if self._superbatch_size > 1:
            labels = torch.cat([t[0] for t in self._superbatch_tups])
            output = torch.cat([t[1] for t in self._superbatch_tups])
            internal_loss = sum(t[2] for t in self._superbatch_tups)
            # print(labels, labels.size())
            # print(output, output.size())
            # print(internal_loss)
        else:
            labels = self.labels
            output = self.output
            internal_loss = self.get_internal_loss()

        if self.loss_module is not None:
            print("Calculating loss: ", self.loss_module.__class__.__name__)
            self.loss_seg = self.loss_module(output, labels) + internal_loss
        else:
            self.loss_seg = F.nll_loss(output, labels) + internal_loss

        print("Doing loss backwards...")
        self.loss_seg.backward()  # calculate gradients of network G w.r.t. loss_G
        # self.loss_seg = self.lossModule(self.output, self.labels)
        # self.loss_seg.backward()


class PointNet2_MP(Segmentation_MP):
    """ Message passing version of PN2"""
