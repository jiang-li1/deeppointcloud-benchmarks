import os.path as osp
import sys
ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
sys.path.append(ROOT)

import open3d as o3d

from utils.custom_datasets.ahn_pointcloud import AHNPointCloud


datapath = '/home/tristan/data/'

cloud = '37EN2_11_section.laz'

pcd = AHNPointCloud.from_cloud(datapath + cloud)

o3dpcd = o3d.geometry.PointCloud()
o3dpcd.points = o3d.utility.Vector3dVector(pcd.pos)




def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()