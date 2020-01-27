import os.path as osp
PARENT_DIR = osp.join(osp.dirname(osp.realpath(__file__)))
ROOT = osp.join(PARENT_DIR, '..', '..')

from pathlib import Path
import json

import numpy as np
import pdal 


def file_to_numpy(fname, to_cache=True, from_cache=True):
    '''
        Function for reading any pointcloud file and converting to numpy array. The array may be a structured array, and may have additional features beyond x, y and z. 
    '''
    path = Path(fname)
    cacheFile = osp.join(PARENT_DIR, '.raw_pointcloud_cache', path.stem) + '.npy'

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
        numpy_to_pdal_writer(narr, 'writers.e57', osp.join(ROOT, 'outputs', 'pointclouds', name + fileType))
    elif fileType == '.laz':
        numpy_to_pdal_writer(narr, 'writers.las', osp.join(ROOT, 'outputs', 'pointclouds', name + fileType))

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

def file_to_recarray(cloud, useCache):
    return file_to_numpy(cloud, from_cache=useCache)


def cloud_to_recarray(cloud, useCache = True):
    if type(cloud) is str:
        return file_to_recarray(cloud, useCache)
    else:
        raise ValueError("Cannot create a PointCloud from a cloud of type {}".format(type(cloud)))