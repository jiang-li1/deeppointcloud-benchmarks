
import torch
import torch_geometric

class PointCloud():
    '''Provides a unified interface, and utility functions, for pointclouds which 
    are stored as torch.tensors

    '''

    def __init__(self, pos: torch.tensor, features: torch.tensor = torch.tensor()):
        self._pos = pos
        self._features = features
        self._minPoint = None
        self._maxPoint = None

        assert self._pos is not None

    @classmethod
    def from_data(cls, data: torch_geometric.data.Data):
        return cls(data.pos, data.x)

    @property
    def pos(self) -> torch.tensor:
        return self._pos

    @property
    def features(self) -> torch.tensor:
        return self._features

    @property
    def minPoint(self) -> torch.tensor:
        if self._minPoint is None:
            self.get_bounding_box()
        return self._minPoint

    @property
    def maxPoint(self) -> torch.tensor:
        if self._maxPoint is None:
            self.get_bounding_box()
        return self._maxPoint

    def get_bounding_box(self):
        minPoint = self.pos.min(dim=0)
        maxPoint = self.pos.max(dim=0)

        self._minPoint = minPoint.values
        self._maxPoint = maxPoint.values

        return self._minPoint, self._maxPoint

    def __len__(self):
        return self.pos.shape[0]

class ClassifiedPointCloud(PointCloud):

    def __init__(self, pos, classes, features=None):
        super().__init__(pos, features)

        self._classes = classes

    @classmethod
    def from_data(cls, data: torch_geometric.data.Data):
        return cls(data.pos, data.y, data.x)

    @property
    def classes(self) -> torch.tensor:
        return self._classes