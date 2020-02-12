
from multiprocessing import Process

import open3d as o3d
import torch 
import numpy as np

from utils.visualization.pcd_utils import *
from src.data.pointcloud import ClassifiedPointCloud, PointCloud

def visualize_subcloud_knn(cloud: PointCloud, sub_idx=None, edge_index=None, window_name="PointCloud"):

    pcd = pointcloud_to_z_grey_o3d_pcd(cloud)
    np.asarray(pcd.colors)[sub_idx] *= np.array([1, 0, 0])

    dest, src = edge_index
    pt_colours = np.random.random((len(cloud), 3))
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(cloud.pos[sub_idx]),
        lines=o3d.utility.Vector2iVector(edge_index.transpose(0, 1)),
    )
    line_set.colors = o3d.utility.Vector3dVector(pt_colours[src])

    vis_non_blocking(pcd, geometries=[line_set], window_name=window_name)


def visualize_subcloud(cloud: PointCloud, sub_idx=None, window_name="PointCloud"):

    pcd = pointcloud_to_z_grey_o3d_pcd(cloud)
    np.asarray(pcd.colors)[sub_idx] *= np.array([1, 0, 0])
    vis_non_blocking(pcd, window_name=window_name)

def visualize_cloud(cloud: PointCloud, window_name='PointCloud'):

    pcd = pointcloud_to_z_grey_o3d_pcd(cloud)
    vis_non_blocking(pcd, window_name=window_name)

def visualize_classes(cloud: ClassifiedPointCloud):

    pcd = clas_pointcloud_to_z_coloured_o3d_pcd(cloud, cloud.classes)

    # o3d.visualization.draw_geometries([pcd])

    vis_non_blocking(pcd, window_name='Classes')

def visualize_predictions(cloud: ClassifiedPointCloud, pred: torch.tensor, inner_idx = None):

    if inner_idx is not None:
        fullPred = torch.full((len(cloud),), -1).to(torch.long)
        fullPred[inner_idx] = pred
        pcd = clas_pointcloud_to_z_coloured_o3d_pcd(cloud, fullPred)
    else:
        pcd = clas_pointcloud_to_z_coloured_o3d_pcd(cloud, pred)

    # o3d.visualization.draw_geometries([pcd])

    vis_non_blocking(pcd, window_name='Predictions')

def visualize_difference(cloud: ClassifiedPointCloud, pred: torch.tensor, inner_idx = None):

    import pdb; pdb.set_trace()
    if inner_idx is not None:
        mask = cloud.classes[inner_idx] == pred
    else:
        mask = cloud.classes == pred

    classes = torch.full((len(cloud),), -1).to(torch.long)

    if inner_idx is not None:
        classes[inner_idx[mask]] = 1
        classes[inner_idx[~mask]] = 0
    else:
        classes[mask] = 1
        classes[~mask] = 0

    pcd = clas_pointcloud_to_z_coloured_o3d_pcd(cloud, classes)

    vis_non_blocking(pcd, window_name='Differences')

    # pcd.estimate_normals()

    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)

    # p = Process(target=o3d.visualization.draw_geometries, args=([pcd, frame],), daemon=True)
    # p.start()

    # o3d.visualization.draw_geometries([pcd, frame])