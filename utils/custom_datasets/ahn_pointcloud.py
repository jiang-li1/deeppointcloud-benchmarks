import os.path as osp
import sys
PARENT_DIR = osp.join(osp.dirname(osp.realpath(__file__)))
sys.path.append(PARENT_DIR)

from overrides import overrides
import numpy as np 
import torch 
from torch_geometric.data import Data

from file_pointcloud import FilePointCloud

def get_log_clip_intensity(intensity):
    return np.log(
        intensity.clip(0.001, 5000)
    )

class RawAHNPointCloud(FilePointCloud):
    '''
        AHNPointCloud with no processing applied. 
    '''

    _feature_names = [
        'PointSourceId', 
        'UserData', 
        'ScanAngleRank', 
        'NumberOfReturns',
        'ReturnNumber',
        'GpsTime',
        'Intensity',
        'Classification'
    ]

    def __init__(self, pos, features, name, recarray):
        super().__init__(pos, name)

        self.features = features
        self.recarray = recarray

    @classmethod
    @overrides
    def from_recarray(cls, recarray, name):

        pos = np.vstack([recarray[field] for field in ['X', 'Y', 'Z']]).T
        features = {f: np.expand_dims(recarray[f], axis=1) for f in cls._feature_names}

        return cls(pos, features, name, recarray)

    @classmethod
    @overrides
    def from_cache(cls, recarray, name):
        raise NotImplementedError

    @overrides
    def to_recarray(self):
        return self.recarray

    @property
    def clas(self):
        return self.features['Classification']



class AHNPointCloud(FilePointCloud):

    _clasNumToName = {
        0 : "other",
        1 : "ground",
        2 : "building",
        3 : "water",
        4 : "infrastructure"
    }

    features = ['intensity', 'num_returns', 'return_ordinal']

    def __init__(self, pos, intensity, num_returns, return_ordinal, clas, name = None):
        super().__init__(pos, name)

        self.intensity = intensity
        self.num_returns = num_returns
        self.return_ordinal = return_ordinal
        self.clas = clas


    @classmethod
    @overrides
    def from_recarray(cls, recarray, name):

        fields = cls.extract_recarray_fields(recarray)
        pcd = cls(*fields, name)

        pcd.log_clip_intensity()
        pcd.remap_classification()

        return pcd

    @classmethod
    @overrides
    def from_cache(cls, recarray, name):
       
        fields = cls.extract_recarray_fields(recarray)
        pcd = cls(*fields, name)

        return pcd

    @overrides
    def to_recarray(self):

        dtypeDict = {
            'X': np.float,
            'Y': np.float,
            'Z': np.float,
            'Intensity': np.float,
            'NumberOfReturns': 'u1',
            'ReturnNumber': 'u1',
            'Classification': 'u1',
        } #only works in python3.6 onwards, where dicts are ordered by default

        recarr = np.rec.fromarrays(
            [
                self.pos[:,0],
                self.pos[:,1],
                self.pos[:,2],
                self.intensity.squeeze(), 
                self.num_returns.squeeze(), 
                self.return_ordinal.squeeze(), 
                self.clas.squeeze()
            ],
            dtype=list(dtypeDict.items())
        )

        return recarr

    @classmethod
    def extract_recarray_fields(cls, recarray):
        pos = np.vstack([recarray[field] for field in ['X', 'Y', 'Z']]).T
        intensity = np.expand_dims(recarray['Intensity'], axis=1)
        num_returns = np.expand_dims(recarray['NumberOfReturns'], axis=1)
        return_ordinal = np.expand_dims(recarray['ReturnNumber'], axis=1)
        clas = np.expand_dims(recarray['Classification'], axis=1)

        return (pos, intensity, num_returns, return_ordinal, clas)
        
    @classmethod
    def clasNumToName(cls):
        return cls._clasNumToName

    @classmethod
    def clasNameToNum(cls):
        return {v: k for k, v in cls.clasNumToName().items()}

    @classmethod
    def clasNames(cls):
        return list(cls.clasNameToNum().keys())

    def to_torch_data(self) -> Data:
        return Data(
            pos = torch.tensor(self.pos, dtype=torch.float32), 
            x = torch.tensor(np.concatenate((self.intensity, self.num_returns, self.return_ordinal), axis=1), dtype=torch.float32),
            y = torch.tensor(self.clas, dtype=torch.long).squeeze(),
        )

    def log_clip_intensity(self):
        self.intensity = np.log(
            self.intensity.clip(0.001, 5000)
        )

    def remap_classification(self):
        self.clas[self.clas == 1] = 0
        self.clas[self.clas == 2] = 1
        self.clas[self.clas == 6] = 2
        self.clas[self.clas == 9] = 3
        self.clas[self.clas == 26] = 4

    def get_points_in_clas(self, clasName):
        index = self.clas == self.clasNameToNum()[clasName]
        index = index.squeeze()

        return AHNPointCloud(self.pos[index], self.intensity[index], self.num_returns[index], self.return_ordinal[index], self.clas[index], self.name + '_' + clasName)

    def split_to_classes(self):

        return {clasName: self.get_points_in_clas(clasName) for clasName in self.clasNameToNum().keys()}

class AHNVehiclePointCloud(AHNPointCloud):
    '''
        AHNPointCloud with vehicles labelled
    '''

    _vclasNumToName = {
        5: "vehicle"
    }

    @classmethod
    def clasNumToName(cls):
        return {**super().clasNumToName(), **cls._vclasNumToName}

    def remap_classification(self):
        super().remap_classification()
        self.clas[self.clas == 27] = 5

