import os.path as osp
import sys
ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
sys.path.append(ROOT)

import open3d as o3d
from utils.visualization.pcd_utils import *
from utils.custom_datasets.file_pointcloud import FilePointCloud

def visualize_file_pointcloud(cloud: FilePointCloud):
    pcd = file_pointcloud_to_o3d_pcd(cloud)
    o3d.visualization.draw_geometries([pcd])

