
import os.path as osp
import sys
ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
sys.path.append(ROOT)

import numpy as np 
import open3d as o3d

from datasets.base_patch_dataset import BasePointCloud

def pointcloud_to_o3d_pcd(cloud : BasePointCloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.pos.numpy())
    return pcd

def pointcloud_to_z_grey_o3d_pcd(cloud: BasePointCloud):
    pcd = pointcloud_to_o3d_pcd(cloud)

    posZScale = (cloud.maxPoint[2].item() - cloud.pos[:,2]) / (cloud.maxPoint[2].item() - cloud.minPoint[2].item())

    colours = np.concatenate(
        (
            np.expand_dims(posZScale, 1),
            np.expand_dims(posZScale, 1),
            np.expand_dims(posZScale, 1),
        ),
        axis=1
    )

    pcd.paint_uniform_color([0.5]*3)
    np.asarray(pcd.colors)[:] = colours
    return pcd

def rect_to_o3d_lineset(minXY, maxXY, minZ=0, maxZ=1, colour=[1, 0, 0]):
    return block_to_o3d_lineset(
        (*minXY, minZ),
        (*maxXY, maxZ),
        colour=colour,
    )

def block_to_o3d_lineset(minXYZ, maxXYZ, colour=[1, 0, 0]):
    pointsIdx = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]

    points = []
    for corner in pointsIdx:
        points.append(
            [
                (minXYZ if minOrMax == 0 else maxXYZ)[dim] 
                for dim, minOrMax in enumerate(corner)
            ]
        )

    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [colour for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set


def visualize_pointcloud(cloud):
    pcd = pointcloud_to_o3d_pcd(cloud)
    o3d.visualization.draw_geometries([pcd])