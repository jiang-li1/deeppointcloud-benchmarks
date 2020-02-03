import os.path as osp
import os
import sys

ROOT = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..")
sys.path.append(ROOT)

import numpy as np
import open3d as o3d
import time
import scipy 
from utils.custom_datasets.ahn_pointcloud import AHNPointCloud, AHNVehiclePointCloud, RawAHNPointCloud, get_log_clip_intensity

from utils.visualization.pcd_utils import colour_z_grey, colour_by_feature

datadir = '/home/tristan/data/raw_car_tiles'
outputdir = '/home/tristan/data/labelled_car_tiles'
patchesPerTile = 2

carTileFiles = [fn for fn in os.listdir(datadir) if fn.lower().endswith('.laz')]

tile1Files = sorted(carTileFiles)[:100]
tile2files = sorted(carTileFiles)[100:]

for fname in tile1Files[:patchesPerTile] + tile2files[:patchesPerTile]:
    raw_pcd = RawAHNPointCloud.from_cloud(datadir + '/' + fname)

    idx = (raw_pcd.clas == 1).squeeze()
    idx = np.asarray([True] * len(raw_pcd.pos))
    pcd = raw_pcd.to_o3d_pcd(idx)



    pcd.estimate_normals()
    colour_z_grey(pcd)

    pcd = pcd.rotate(
        scipy.spatial.transform.Rotation.from_rotvec([0, 1, 0]).as_matrix()
    )
    # colour_by_feature(pcd, get_log_clip_intensity(raw_pcd.features['Intensity'][idx]))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().load_from_json('ro.json')

    vis.add_geometry(pcd)
    vis.run()
    # o3d.visualization.draw_geometries([pcd])

