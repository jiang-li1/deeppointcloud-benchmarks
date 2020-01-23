
import open3d as o3d
import torch 

from utils.visualization.pcd_utils import *
from src.datasets.base_patch_dataset import Grid2DPatchDataset, ClassifiedPointCloud

def visualize_classes(cloud: ClassifiedPointCloud):

    pcd = clas_pointcloud_to_z_coloured_o3d_pcd(cloud, cloud.classes)

    o3d.visualization.draw_geometries([pcd])

def visualize_predictions(cloud: ClassifiedPointCloud, pred: torch.tensor):

    pcd = clas_pointcloud_to_z_coloured_o3d_pcd(cloud, pred)

    o3d.visualization.draw_geometries([pcd])

def visualize_difference(cloud: ClassifiedPointCloud, pred: torch.tensor):
    
    mask = cloud.classes == pred

    classes = torch.zeros((len(cloud),)).to(torch.long)

    classes[mask] = -1
    classes[~mask] = 0

    pcd = clas_pointcloud_to_z_coloured_o3d_pcd(cloud, classes)

    o3d.visualization.draw_geometries([pcd])