import os.path as osp
import sys

ROOT = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..")
sys.path.append(ROOT)

import numpy as np
import open3d as o3d
import time
from utils.custom_datasets.ahn_pointcloud import AHNPointCloud
from utils.visualization.pcd_utils import colour_z_grey

# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

datapath = "/home/tristan/deeppointcloud-benchmarks/data/AHNTilesDataset/raw/"

cloud = "37EN2_11_section_tiny.laz"

pcd = AHNPointCloud.from_cloud(datapath + cloud)

global o3dpcd

o3dpcd = o3d.geometry.PointCloud()
o3dpcd.points = o3d.utility.Vector3dVector(pcd.pos)
o3dpcd.estimate_normals()
o3dpcd.paint_uniform_color([0.5] * 3)
# o3dpcd = colour_z_grey(o3dpcd)

vis = o3d.visualization.VisualizerWithEditing()
# vis = o3d.visualization.VisualizerWithVertexSelection()
vis.create_window()
vis.add_geometry(o3dpcd)


def anim_cb(v: o3d.visualization.Visualizer):
    global o3dpcd
    pass
    idx = v.get_selected_points()
    # idx = [p.index for p in v.get_picked_points()]

    if len(idx) > 0:
        np.asarray(o3dpcd.colors)[idx] *= np.array([1, 0, 0])

        # v.update_geometry(o3dpcd)
        v.clear_geometries()
        # v.remove_geometry(o3dpcd, False)
        v.add_geometry(o3dpcd, False)

        # v.update_renderer()
        # print(np.asarray(o3dpcd.colors)[idx])
        return True

    return False


vis.register_animation_callback(anim_cb)

vis.run()
vis.destroy_window()

# print('starting vis loop')
# start = time.time()
# while time.time() - start < 10:
#     vis.update_geometry(o3dpcd)
#     vis.poll_events()
#     vis.update_renderer()

# print('vis loop ended')


def demo_crop_geometry(pcd):
    print("Demo for manual geometry cropping")
    print("1) Press 'Y' twice to align geometry with negative direction of y-axis")
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")
    o3d.visualization.draw_geometries_with_editing([pcd])


def pick_points(pcd):
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()
