
import os.path as osp
import sys
ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..')
sys.path.append(ROOT)

import open3d as o3d 

from base_patch_dataset import BasePointCloud

def build_kdtree(cloud : BasePointCloud):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.pos.numpy())
    return o3d.geometry.KDTreeFlann(cloud)