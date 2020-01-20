
import hydra

import os.path as osp
import sys
ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
sys.path.append(ROOT)

import open3d as o3d
import numpy as np

from pcd_utils import visualize_pointcloud, pointcloud_to_o3d_pcd, rect_to_o3d_lineset
from datasets.utils import find_dataset_using_name
from datasets.base_patch_dataset import Grid2DPatchDataset


@hydra.main(config_path = osp.join(ROOT, 'conf', 'config.yaml'))
def main(cfg):
    
    dataset_name = cfg.experiment.dataset
    dataset_config = cfg.data[dataset_name]

    dataset = find_dataset_using_name(dataset_name)(dataset_config, cfg.training)

    # visualize_pointcloud(dataset.train_dataset._cloud_dataset[0])

    patchDataset = dataset.train_dataset.patch_datasets[0]
    visualize_2d_grid_dataset(patchDataset)

def visualize_2d_grid_dataset(dataset: Grid2DPatchDataset):

    pcd = pointcloud_to_o3d_pcd(dataset)

    global tileIdx
    tileIdx = dataset.get(0).detach().numpy()

    pcd.paint_uniform_color([0.5]*3)

    pos = dataset.pos.detach().numpy()
    posZScale = (dataset.maxPoint[2].item() - pos[:,2]) / (dataset.maxPoint[2].item() - dataset.minPoint[2].item())

    colours = np.concatenate(
        (
            np.expand_dims(posZScale, 1),
            np.expand_dims(posZScale, 1),
            np.expand_dims(posZScale, 1),
        ),
        axis=1
    )

    np.asarray(pcd.colors)[:] = colours


    minZ = dataset.pos[tileIdx].min(dim=0).values[2].item()
    maxZ = dataset.pos[tileIdx].max(dim=0).values[2].item()

    # colours = np.tile([1, 0, 0], (tileIdx.shape[0], 1))
    tilePts = dataset.pos[tileIdx,2].detach().numpy()

    zScale = (maxZ - tilePts) / (maxZ - minZ)

    tileColours = np.concatenate(
        (
            np.expand_dims(zScale, 1),
            np.zeros((zScale.shape[0], 1)), 
            np.zeros((zScale.shape[0], 1))
        ),
        axis=1
    )

    np.asarray(pcd.colors)[tileIdx] = tileColours
    lineset = rect_to_o3d_lineset(
        *dataset._get_bounds_for_idx(0), 
        dataset.minPoint[2], 
        dataset.maxPoint[2]
    )

    global i
    i = 1

    def vis_callback(vis):

        global tileIdx, i
        np.asarray(pcd.colors)[tileIdx] = colours[tileIdx]
        
        tileIdx = dataset.get(i).detach().numpy()
        tilePts = dataset.pos[tileIdx,2].detach().numpy()
        minZ = dataset.pos[tileIdx].min(dim=0).values[2].item()
        maxZ = dataset.pos[tileIdx].max(dim=0).values[2].item()
        zScale = (maxZ - tilePts) / (maxZ - minZ)
        tileColours = np.concatenate(
            (
                np.expand_dims(zScale, 1),
                np.zeros((zScale.shape[0], 1)), 
                np.zeros((zScale.shape[0], 1))
            ),
            axis=1
        )
        np.asarray(pcd.colors)[tileIdx] = tileColours

        i += 1
        vis.update_geometry(pcd)

    o3d.visualization.draw_geometries_with_key_callbacks(
        [pcd, lineset], 
        {ord(' '): vis_callback}
    )

    o3d.visualization.draw_geometries([pcd, lineset])

if __name__ == '__main__':
    main()