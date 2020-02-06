
import torch
import torch_geometric

from src.data.patch.base_patchable_pointcloud import BasePatchablePointCloud

class PointBallPatchableCloud(BasePatchablePointCloud):
    '''
        Base class for patch datasets which return balls of points centered on
        points in the point clouds
    '''

    def __init__(self, data: torch_geometric.data.Data, radius, context_dist):
        super().__init__(data)

        self._radius = radius
        self._context_dist = context_dist

    def __len__(self):
        return len(self.data.pos)

    def __getitem__(self, idx):

        center = self.pos[idx]

        patch_idx = torch.dist(center, self.pos) < self._radius

        d = torch_geometric.data.Data(
            pos = self.pos[patch_idx],
            x = self.features[patch_idx],
            y = self.data.y[patch_idx]
        )

        return d

# class BallPatchableCloud(BasePatchablePointCloud):

#     def __init__(self, pos):
#         super().__init__(pos)

#         self.kdtree = build_kdtree(self)

#     def radius_query(self, point: torch.tensor, radius):
#         k, indices, dist2 = self.kdtree.search_radius_vector_3d(point, radius)
#         return k, indices, dist2

#     def knn_query(self, point: torch.tensor, k):
#         k, indices, dist2 = self.kdtree.search_knn_vector_3d(point, k)
#         return k, indices, dist2