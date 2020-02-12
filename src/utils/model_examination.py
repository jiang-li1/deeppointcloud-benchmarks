
from typing import List, Union

import torch 

from src.models.base_model import BaseModel
from src.core.common_modules import BaseModule

def get_modules_of_type(model: BaseModel, types: Union[type, List[type]]):

    if type(types) == type:
        typesList = [types]
    else:
        typesList = types

    return _search_children(model, typesList)

def _search_children(module: BaseModule, types: List[type]):

    modules = []

    for child in module.children():
        if type(child) in types:
            modules.append(child)
        modules += _search_children(child, types)

    return modules

class LayerInfo:

    def __init__(self, pos, samp_idx, edge_index):

        self.pos = pos
        self.samp_idx = samp_idx
        self.edge_index = edge_index
