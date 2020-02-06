import os
import os.path as osp
import sys
import itertools
import functools
import pathlib

import torch
import torch_geometric
from torch.utils.data import RandomSampler
from torch_geometric.data import InMemoryDataset, Dataset

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..")
sys.path.append(ROOT)

from src.data.patch.patch_dataset import PatchDataset, PartialPatchDataset
from src.data.patch.grid2D_patchable_cloud import Grid2DPatchableCloud
from src.data.falible_dataset import FalibleDatasetWrapper, FalibleIterDatasetWrapper
from src.data.sampler import UniqueRandomSampler, UniqueSequentialSampler
from src.data.base_dataset import BaseDataset

from src.metrics.ahn_tracker import AHNTracker

from src.datasets.utils.downloader import download_url
from src.datasets.utils.las_splitter import split_las_pointcloud

from utils.custom_datasets.ahn_pointcloud import AHNPointCloud

class AHNSubTileDataset(Dataset):
    '''Dataset backed by a set of tiles from AHN, which exposes 
    each tile as a set of sub-tiles. __len__ returns the number 
    of subtiles, and get(idx) returns the idx^th subtile. 

        Download tile from pdok
                |
                v
        Split into subtiles using LAStools
                |
                v
        Use AHNPointCloud class to process each subtile
        and convert it into a torch Data object
    '''

    train_tiles = [
        '38FN1',
        '31HZ2',
    ]

    test_tiles = [
        '32CN1',
        '37EN2',
    ]

    num_subtiles = 16 #num_subtiles^0.5 must be an integer

    def __init__(self, root, split, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        if self.split == 'train':
            return self.get_subtile_names(self.train_tiles, 'LAZ')
        elif self.split == 'test':
            return self.get_subtile_names(self.test_tiles, 'LAZ')
        else:
            raise ValueError('Split {} not recognized'.format(self.split))

    # @property
    # def processed_file_names(self):
    #     return ['{}_num_subtiles={}_data.pt'.format(
    #         '_'.join(self.raw_file_names),
    #         self.num_subtiles
    #     )]

    @property
    def processed_file_names(self):
        if self.split == 'train':
            return self.get_subtile_names(self.train_tiles, 'pt')
        elif self.split == 'test':
            return self.get_subtile_names(self.test_tiles, 'pt')
        else:
            raise ValueError('Split {} not recognized'.format(self.split))

    def __len__(self):
        return len(self.processed_file_names)

    def get(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        print('wid: {} loading file:'.format(worker_info.id if worker_info else -1), self.processed_paths[idx])
        data = torch.load(self.processed_paths[idx])
        data.name = self.processed_file_names[idx].split('.')[0]
        return data

    def download(self):
        url = 'https://geodata.nationaalgeoregister.nl/ahn3/extract/ahn3_laz/C_{}.LAZ'

        tilesList = self.train_tiles if self.split == 'train' else self.test_tiles

        for tilename in tilesList:
            filename = download_url(url.format(tilename), self.root + '/tiles/')
            split_las_pointcloud(filename, self.num_subtiles, self.root + '/raw/')


    def process(self):
        # data_list = []

        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):

            print('Converting laz to torch.data: {} -> {}'.format(raw_path, processed_path))

            if osp.exists(processed_path):
                print('{} exists. skipping conversion'.format(processed_path))
            else:
                pcd = AHNPointCloud.from_cloud(raw_path)
                torch.save(pcd.to_torch_data(), processed_path)

        # for raw_path in self.raw_paths:
        #     pcd = AHNPointCloud.from_cloud(raw_path)
        #     # tracker.print_diff()
        #     data_list.append(pcd.to_torch_data())
        #     del pcd
        #     gc.collect()


        # torch.save(self.collate(data_list), self.processed_paths[0])


    def get_subtile_names(self, tile_names, suffix):
        return list(itertools.chain(*[
            ['C_' + tn + '_{}.{}'.format(i, suffix) for i in range(self.num_subtiles)]
            for tn in tile_names
        ]))

class AHNTilesDataset(InMemoryDataset):

    adriaan_tiles_train = [
            '37EN2_11.LAZ',
            '37EN2_16.LAZ',
            '43FZ2_20.LAZ',
    ]
    adriaan_tiles_test = [
            '37FZ2_21.LAZ',

    ]

    small_tile = [
        '37EN2_11_section.laz'
    ]

    tiny_tile = [
        '37EN2_11_section_tiny.laz'
    ]


    def __init__(self, root, split, transform=None, pre_transform=None, pre_filter=None):

        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.split == 'eval':
            return self.adriaan_tiles_test
        return self.adriaan_tiles_train if self.split == 'train' else self.adriaan_tiles_test
        # return self.small_tile
        
    @property
    def processed_file_names(self):
        return ['{}_data.pt'.format('_'.join(self.raw_file_names))]

    def download(self):
        raise NotImplementedError

    def process(self):

        data_list = []

        for raw_path in self.raw_paths:
            pcd = AHNPointCloud.from_cloud(raw_path)

            data_list.append(pcd.to_torch_data())

        torch.save(self.collate(data_list), self.processed_paths[0])

class AHNGridPatchableCloud(Grid2DPatchableCloud):

    def __init__(self, data, patch_diam=10, context_dist=0.3, eval_mode=False, *args, **kwargs):
        super().__init__(data, patch_diam, patch_diam, context_dist, eval_mode, *args, **kwargs)

        self.num_classes = 5

    @classmethod
    def from_tiles_dataset(cls, dataset: AHNTilesDataset, **kwargs):
        assert len(dataset) == 1
        return cls(dataset[0], **kwargs)

# class AHNMultiCloudPatchDataset(BaseMultiCloudPatchDataset):

#     def __init__(self, patchDatasets):
#         super().__init__(patchDatasets)

#         self.num_classes = 5

#     @classmethod
#     def from_backing_dataset(cls, backingDataset, patch_opt):
#         return cls([
#             AHNPatchDataset(data, **patch_opt) for data in backingDataset
#         ])


# class AHNLargeMultiCloudPatchDataset(BaseLargeMultiCloudPatchDataset):

#     def __init__(self, root, split, dataset_opt):
#         def make_patch_dataset(data: torch.utils.data.Data) -> AHNPatchDataset:
#             return AHNPatchDataset(data, **dataset_opt.patch_opt)

#         def make_mc_patch_dataset(patchDatasets: List[AHNPatchDataset]) -> BaseMultiCloudPatchDataset:
#             mcpd = BaseMultiCloudPatchDataset
#             mcpd.num_classes = 5
#             return mcpd

#         super().__init__(
#             AHNSubTileDataset(root, split),
#             make_mc_patch_dataset,
#             make_patch_dataset,
#             dataset_opt.samples_per_subtile,
#             dataset_opt.num_loaded_subtiles,
#         )


class AHNAerialDataset(BaseDataset):

    def __init__(self, dataset_opt, training_opt, eval_mode=False):
        super().__init__(dataset_opt, training_opt)
        self._data_path = osp.join(ROOT, dataset_opt.dataroot, "AHNSubTilesDataset")

        def make_patchable_cloud(data: torch_geometric.data.Data) -> AHNGridPatchableCloud:
            return AHNGridPatchableCloud(data, **dataset_opt.patch_opt)

        train_tiles_dataset = AHNSubTileDataset(self._data_path, "train")
        train_patch_dataset = PartialPatchDataset(
            train_tiles_dataset,
            make_patchable_cloud,
            datasets_per_worker=1,
            samples_per_dataset=20
        )
        self.train_dataset = FalibleDatasetWrapper(
            train_patch_dataset,
            UniqueSequentialSampler(train_patch_dataset, training_opt.num_workers)
        )

        test_tiles_dataset = AHNSubTileDataset(self._data_path, "test")
        test_patch_dataset = PartialPatchDataset(
            test_tiles_dataset,
            make_patchable_cloud,
            datasets_per_worker=1,
            samples_per_dataset=20
        )
        self.test_dataset = FalibleDatasetWrapper(
            test_patch_dataset, 
            UniqueSequentialSampler(test_patch_dataset, training_opt.num_workers)
        )

        self.pointcloud_scale = dataset_opt.scale

        self._create_dataloaders(
            self.train_dataset,
            self.test_dataset,
            val_dataset=None,
            num_test_workers=training_opt.num_workers//2
        )

    @property
    def class_num_to_name(self):
        return AHNPointCloud.clasNumToName()

    @property
    def class_to_segments(self):
        return {
            k: [v] for k, v in AHNPointCloud.clasNameToNum().items()
        }

    @staticmethod
    def get_tracker(model, task: str, dataset, wandb_opt: bool, tensorboard_opt: bool):
        """Factory method for the tracker

        Arguments:
            task {str} -- task description
            dataset {[type]}
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        return AHNTracker(dataset, wandb_log=wandb_opt.log, use_tensorboard=tensorboard_opt.log)


class AHNSmallDataset(BaseDataset):

    def __init__(self, dataset_opt, training_opt, eval_mode=False):
        super().__init__(dataset_opt, training_opt)
        self._data_path = osp.join(ROOT, dataset_opt.dataroot, "AHNTilesDataset")

        if eval_mode:
            self._init_for_eval(dataset_opt, training_opt)
        else:

            train_tiles_dataset = AHNTilesDataset(self._data_path, "train")
            train_patch_dataset = PatchDataset(
                [AHNGridPatchableCloud(d, **dataset_opt.patch_opt) for d in train_tiles_dataset]
            )
            self.train_dataset = FalibleDatasetWrapper(
                train_patch_dataset,
                UniqueRandomSampler(
                    train_patch_dataset,
                    replacement=True,
                    num_samples=100//training_opt.num_workers, #sample ~100 patches in total per epoch
                )
            )
            test_tiles_dataset = AHNTilesDataset(self._data_path, "test")
            test_patch_dataset = PatchDataset(
                [AHNGridPatchableCloud(d, **dataset_opt.patch_opt) for d in test_tiles_dataset]
            )
            self.test_dataset = FalibleDatasetWrapper(
                test_patch_dataset,
                UniqueRandomSampler(
                    test_patch_dataset,
                    replacement=True,
                    num_samples=50//training_opt.num_workers,
                )
            )


            # self._create_dataloaders(
            #     self.train_dataset, 
            #     self.test_dataset, 
            #     validation=None,
            #     train_sampler=RandomSampler(
            #         self.train_dataset, 
            #         replacement=True,
            #         num_samples=100
            #     ),
            #     test_sampler=RandomSampler(
            #         self.test_dataset,
            #         replacement=True,
            #         num_samples=50
            #     )
            # )
            self._create_dataloaders(
                self.train_dataset,
                self.test_dataset,
                val_dataset=None,
            )

        self.pointcloud_scale = dataset_opt.scale

    def _init_for_eval(self, dataset_opt, training_opt):

        test_tiles_dataset = AHNTilesDataset(self._data_path, "eval")
        test_patch_dataset = PatchDataset(
            [AHNGridPatchableCloud(test_tiles_dataset, **dataset_opt.patch_opt, eval_mode=True) for d in test_tiles_dataset]

        )
        self.test_dataset = FalibleDatasetWrapper(
            test_patch_dataset,
            UniqueSequentialSampler(
                training_opt.num_workers,
                test_patch_dataset
            )
        )

        self._create_dataloaders(
            self.test_dataset,
            self.test_dataset,
            validation=None
        )

    @property
    def class_num_to_name(self):
        return AHNPointCloud.clasNumToName()

    @property
    def class_to_segments(self):
        return {
            k: [v] for k, v in AHNPointCloud.clasNameToNum().items()
        }

    @staticmethod
    def get_tracker(model, task: str, dataset, wandb_opt: bool, tensorboard_opt: bool):
        """Factory method for the tracker

        Arguments:
            task {str} -- task description
            dataset {[type]}
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        return AHNTracker(dataset, wandb_log=wandb_opt.log, use_tensorboard=tensorboard_opt.log)

def _test():
    data_path = osp.join(ROOT, 'data', "AHNSubTilesDataset")
    d = AHNSubTileDataset(data_path, 'train')
    return d


