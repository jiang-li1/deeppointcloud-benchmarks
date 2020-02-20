import importlib

import torch 
import etw_pytorch_utils as pt_utils

from src.utils.module_builder import build_module

from .base import Segmentation_MP
from src.modules.RandLANet import *
from src.models.base_architectures import UnetBasedModel
from src.core.base_conv.dense import DenseFPModule

from overrides import overrides

class RandLANetSeg(Segmentation_MP):
    """ Unet base implementation of RandLANet
    """

    def __init__(self, *args, **kwargs):
        super().__init__(superbatch_size=10, *args, **kwargs)

class RandLANetSeg_D(UnetBasedModel):

    def __init__(self, option, model_type, dataset, modules, **kwargs):
        super().__init__(option, model_type, dataset, modules, superbatch_size=10, **kwargs)

        self._num_classes = dataset.num_classes


        # Last MLP
        last_mlp_opt = option.mlp_cls

        self.FC_layer = pt_utils.Seq(last_mlp_opt.nn[0])
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.conv1d(last_mlp_opt.nn[i], bn=True)
        if last_mlp_opt.dropout:
            self.FC_layer.dropout(p=last_mlp_opt.dropout)

        self.FC_layer.conv1d(self._num_classes, activation=None)
        self.loss_names = ["loss_seg"]

        self.loss_module = build_module(option.loss_module, importlib.import_module("src.core.losses.losses"))
    
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

    def forward(self):
        r"""
            Forward pass of the network
            self.input:
                x -- Features [B, C, N]
                pos -- Points [B, N, 3]
        """
        data = self.model(self.input)
        last_feature = data.x

        self.output = self.FC_layer(last_feature).transpose(1, 2).contiguous().view((-1, self._num_classes))

        self.output = F.softmax(self.output, dim=-1)
        
        return self.output
    
    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
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