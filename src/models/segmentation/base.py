import logging
import torch
import torch.nn.functional as F
from typing import Any
import importlib

from src.models.base_architectures import UnetBasedModel

from src.utils.module_builder import build_module

log = logging.getLogger(__name__)


class Segmentation_MP(UnetBasedModel):
    def __init__(self, option, model_type, dataset, modules, **kwargs):
        """Initialize this model class.
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        UnetBasedModel.__init__(
            self, option, model_type, dataset, modules, **kwargs
        )  # call the initialization method of UnetBasedModel
        nn = option.mlp_cls.nn
        self.dropout = option.mlp_cls.get("dropout")
        self.lin1 = torch.nn.Linear(nn[0], nn[1])
        self.lin2 = torch.nn.Linear(nn[2], nn[3])
        self.lin3 = torch.nn.Linear(nn[4], dataset.num_classes)

        self.loss_names = ["loss_seg"]

        if 'loss_module' in option:
            self.loss_module = build_module(option.loss_module, importlib.import_module("src.core.losses.losses"))
        else:
            self.loss_module = None

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input = data
        self.labels = data.y
        self.batch_idx = data.batch
        # self.batch_idx = torch.arange(0, data.pos.shape[0]).view(-1, 1).repeat(1, data.pos.shape[1]).view(-1)

    def forward(self) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        data = self.model(self.input)
        x = F.relu(self.lin1(data.x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        self.output = F.log_softmax(x, dim=-1)
        # self.loss_seg = F.nll_loss(self.output, self.labels) + self.get_internal_loss()
        self.loss_seg = -1
        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results

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
            internal_loss = self.internal_loss

        if self.loss_module is not None:
            self.loss_seg = self.loss_module(output, labels) + internal_loss
        else:
            self.loss_seg = F.nll_loss(output, labels) + internal_loss

        print("Doing loss backwards...")
        self.loss_seg.backward()  # calculate gradients of network G w.r.t. loss_G
