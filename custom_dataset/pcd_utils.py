import os.path as osp
ROOT = osp.join(osp.dirname(osp.realpath(__file__)))
from collections import OrderedDict
import inspect 
from abc import ABC, abstractmethod
import hashlib
from pathlib import Path
import json

import open3d as o3d
import torch
from torch_geometric.data import Data
import numpy as np
import pdal 
import pandas
from overrides import overrides

def file_to_numpy(fname, to_cache=True, from_cache=True):
    '''
        Function for reading any pointcloud file and converting to numpy array. The array may be a structured array, and may have additional features beyond x, y and z. 
    '''
    path = Path(fname)
    cacheFile = osp.join(ROOT, '.raw_pointcloud_cache', path.stem) + '.npy'

    if from_cache:
        if osp.exists(cacheFile):
            print('Using cached cloud: ', cacheFile)
            return np.load(cacheFile)

    extension = path.suffix.lower()
    if extension == '.laz':
        arr = pdal_reader_to_numpy('readers.las', fname)
    else:
        raise NotImplementedError("File extension {} not supported".format(extension))

    if to_cache:
        np.save(cacheFile, arr)

    return arr


def pdal_reader_to_numpy(reader, fname):
    pipeline = [
        {
            "type": reader,
            "filename": fname,
        }
    ]
    jsonStr = json.dumps(pipeline)
    pdalPipeline = pdal.Pipeline(jsonStr)
    pdalPipeline.validate()
    pdalPipeline.execute()
    arrays = pdalPipeline.arrays
    return arrays[0]

def numpy_to_file(narr, name, fileType):

    if fileType == '.e57':
        numpy_to_pdal_writer(narr, 'writers.e57', osp.join(ROOT, 'output_pointclouds', name + fileType))
    elif fileType == '.laz':
        numpy_to_pdal_writer(narr, 'writers.las', osp.join(ROOT, 'output_pointclouds', name + fileType))

def recarray_col_as_type(recarr : np.ndarray, colName, newType):

    dtypeDict = OrderedDict(recarr.dtype.descr)
    dtypeDict[colName] = newType
    newDtype = np.dtype(list(dtypeDict.items()))
    return recarr.astype(newDtype)

def recarry_view_fields(recarr: np.ndarray, fieldsList):
    return recarr.getfield(np.dtype(
        {name: recarr.dtype.fields[name] for name in fieldsList}
    ))


def numpy_to_pdal_writer(narr, writer, fname):
    pipeline = [
        {
            "type": writer,
            "filename": fname,
        }
    ]
    jsonStr = json.dumps(pipeline)
    pdalPipeline = pdal.Pipeline(jsonStr, [narr])
    pdalPipeline.validate()
    pdalPipeline.execute()
    return

def get_log_clip_intensity(arr):
    arr = recarray_col_as_type(arr, 'Intensity', np.float)
    arr['Intensity'] = np.log(
        arr['Intensity'].clip(0, 5000)
    )
    return arr

def remap_classification(arr):
    clas = arr['Classification']
    clas[clas == 6] = 3
    clas[clas == 9] = 4
    clas[clas == 26] = 5

def get_class_id(classObj):
    return classObj.__name__ + '_' + hashlib.sha256(
        inspect.getsource(classObj).encode()
    ).hexdigest()[:10]

def file_to_recarray(cloud, useCache):
    return file_to_numpy(cloud, from_cache=useCache)


def cloud_to_recarray(cloud, useCache = True):
    if type(cloud) is str:
        return file_to_recarray(cloud, useCache)
    else:
        raise ValueError("Cannot create a PointCloud from a cloud of type {}".format(type(cloud)))


class PointCloud(ABC):
    '''
        Base class for pointclouds. Extending classes must define functions to create
        pointclouds from np recarrays and np recarrays from pointclouds. This is because (a) np recarrays can be easily stored as .npy files and (b) pdal is used to convert between PointCloud objects and various file formats - and pdal accepts and returns np recarrays. 

        The class if for representing pointclouds from a given dataset - for all such pointclouds a similar process will need to be used to convet from recarrays to pointcloud objects 
    '''

    def __init__(self, pos, name = None):
        
        self.name = name        
        self.pos = pos
        
    @classmethod
    def from_cloud(cls, cloud, name = None, useCache = True):

        if name is None and type(cloud) is str:
            name = Path(cloud).stem

        cacheFile = osp.join(ROOT, '.processed_pointcloud_cache', name) + '_' + get_class_id(cls) + '.npy'

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
    
    def to_dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame(self.to_recarray())

    def to_o3d_pcd(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pos)
        return pcd

    def to_e57(self):
        numpy_to_file(self.to_recarray(), self.name, '.e57')

    def to_laz(self):
        numpy_to_file(self.to_recarray(), self.name, '.laz')

class AHNPointCloud(PointCloud):

    clasNumToName = {
        1 : "vegetation",
        2 : "ground",
        3 : "building",
        4 : "water",
        5 : "infrastructure"
    }

    features = ['intensity', 'num_returns', 'return_ordinal']

    clasNameToNum = {v: k for k, v in clasNumToName.items()}

    clasNames = list(clasNameToNum.keys())

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
        } #only works in python3.6 onwards

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
        self.clas[self.clas == 6] = 3
        self.clas[self.clas == 9] = 4
        self.clas[self.clas == 26] = 5

    def get_points_in_clas(self, clasName):
        index = self.clas == self.clasNameToNum[clasName]
        index = index.squeeze()

        return AHNPointCloud(self.pos[index], self.intensity[index], self.num_returns[index], self.return_ordinal[index], self.clas[index], self.name + '_' + clasName)

    def split_to_classes(self):

        return {clasName: self.get_points_in_clas(clasName) for clasName in self.clasNameToNum.keys()}






