from .base import Segmentation_MP
from src.modules.RandLANet import *
from overrides import overrides

class RandLANetSeg(Segmentation_MP):
    """ Unet base implementation of RandLANet
    """
