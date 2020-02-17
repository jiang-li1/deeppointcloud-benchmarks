from .base import Segmentation_MP
from src.modules.RandLANet import *
from overrides import overrides

class RandLANetSeg(Segmentation_MP):
    """ Unet base implementation of RandLANet
    """

    def __init__(self, *args, **kwargs):
        super().__init__(superbatch_size=10, *args, **kwargs)
