import os.path as osp
import sys
PARENT_DIR = osp.join(osp.dirname(osp.realpath(__file__)))
ROOT = osp.join(PARENT_DIR, '..', '..')
sys.path.append(ROOT)

from pathlib import Path
from collections import OrderedDict
import hashlib
import inspect
from abc import ABC, abstractmethod

import numpy as np
import pandas


def recarray_col_as_type(recarr : np.ndarray, colName, newType):

    dtypeDict = OrderedDict(recarr.dtype.descr)
    dtypeDict[colName] = newType
    newDtype = np.dtype(list(dtypeDict.items()))
    return recarr.astype(newDtype)

def recarry_view_fields(recarr: np.ndarray, fieldsList):
    return recarr.getfield(np.dtype(
        {name: recarr.dtype.fields[name] for name in fieldsList}
    ))

def get_class_id(classObj):
    return classObj.__name__ + '_' + hashlib.sha256(
        inspect.getsource(classObj).encode()
    ).hexdigest()[:10]


class FilePointCloud(ABC):
    '''
        Base class for pointclouds read from and written to files, 
        such as pointclouds from a dataset. 
        
        Extending classes must define functions to create
        pointclouds from np recarrays and np recarrays from pointclouds. 
        This is because (a) np recarrays can be easily stored as .npy files and 
        (b) pdal is used to convert between PointCloud objects and 
        various file formats - and pdal accepts and returns np recarrays. 

        The class is for representing pointclouds from a given dataset 
        - for all such pointclouds a similar process will need to be used 
        to convert from recarrays to pointcloud objects 
    '''

    def __init__(self, pos, name = None):
        
        self._name = name        
        self._pos = pos
        
    @classmethod
    def from_cloud(cls, cloud, name = None, useCache = True):
        from utils.custom_datasets.pcd_io_utils import cloud_to_recarray

        if name is None and type(cloud) is str:
            name = Path(cloud).stem

        cacheFile = osp.join(PARENT_DIR, '.processed_pointcloud_cache', name) + '_' + get_class_id(cls) + '.npy'

        if useCache and name:
            if osp.exists(cacheFile):
                print('Using cached PointCloud: ', cacheFile)
                arr = np.load(cacheFile)
                return cls.from_cache(arr, name)

        arr = cloud_to_recarray(cloud, useCache)
        pcd = cls.from_recarray(arr, name)

        if useCache:
            np.save(cacheFile, pcd.to_recarray())

        return pcd

    @classmethod
    @abstractmethod
    def from_recarray(cls, recarray, name):
        '''
            Create a PointCloud from a recarray. This might involve selecting a subset 
            of recarray columns, or applying preprocessing on certain columns
        '''
        pass


    @classmethod
    @abstractmethod
    def from_cache(cls, recarray, name):
        pass

    @abstractmethod
    def to_recarray(self):
        pass

    @property
    def pos(self) -> np.ndarray:
        return self._pos

    @property
    def name(self) -> str:
        return self._name
    
    
    def to_dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame(self.to_recarray())

    # def to_o3d_pcd(self) -> o3d.geometry.PointCloud:
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(self.pos)
    #     return pcd

    def to_e57(self):
        from utils.custom_datasets.pcd_io_utils import numpy_to_file
        numpy_to_file(self.to_recarray(), self.name, '.e57')

    def to_laz(self):
        from utils.custom_datasets.pcd_io_utils import numpy_to_file
        numpy_to_file(self.to_recarray(), self.name, '.laz')