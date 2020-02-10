
from multiprocessing import Process

import open3d as o3d
import torch 

from utils.visualization.pcd_utils import *
from src.data.pointcloud import ClassifiedPointCloud

def visualize_classes(cloud: ClassifiedPointCloud):

    pcd = clas_pointcloud_to_z_coloured_o3d_pcd(cloud, cloud.classes)

    # o3d.visualization.draw_geometries([pcd])

    vis_non_blocking(pcd, window_name='Classes')

def visualize_predictions(cloud: ClassifiedPointCloud, pred: torch.tensor):

    pcd = clas_pointcloud_to_z_coloured_o3d_pcd(cloud, pred)

    # o3d.visualization.draw_geometries([pcd])

    vis_non_blocking(pcd, window_name='Predictions')

def visualize_difference(cloud: ClassifiedPointCloud, pred: torch.tensor):
    
    mask = cloud.classes == pred

    classes = torch.zeros((len(cloud),)).to(torch.long)

    classes[mask] = -1
    classes[~mask] = 0

    pcd = clas_pointcloud_to_z_coloured_o3d_pcd(cloud, classes)

    vis_non_blocking(pcd, window_name='Differences')

    # pcd.estimate_normals()

    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)

    # p = Process(target=o3d.visualization.draw_geometries, args=([pcd, frame],), daemon=True)
    # p.start()

    # o3d.visualization.draw_geometries([pcd, frame])