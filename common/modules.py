# Non-Parametric Decoder
import torch
from torch import nn

from libs.lib_utils import index_points, angle


class GeoDecoder(nn.Module):
    def __init__(self, num_stages, de_neighbors, cat=False):
        super().__init__()
        self.num_stages = num_stages
        self.de_neighbors = de_neighbors
        self.cat = cat

    def propagate(self, xyz1, xyz2, normal1, normal2, points1=None, points2=None):
        """
        Input:
            xyz1: input points position, [B, N, 3]
            xyz2: sampled input points position semdata, [B, S, 3]
            normal1: input points normal, [B, N, 3]
            normal2: sampled input point normal, [B, S, 3]
            points1: input points, [B, N, D']
            points2: input points, [B, S, D'']
        Return:
            new_points: upsampled points, [B, D''', N]
        """
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = torch.cdist(xyz1, xyz2).clip(min=0.0) + torch.sigmoid(torch.cdist(normal1, normal2))
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :self.de_neighbors], idx[:, :, :self.de_neighbors]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            weight = weight.view(B, N, self.de_neighbors, 1)
            # index_points(xyz1, idx)
            interpolated_points = torch.sum(index_points(points2.permute(0, 2, 1), idx) * weight, dim=2)

        if (points1 is not None) and isinstance(points1, torch.Tensor):
            if self.cat:
                new_points = torch.cat([points1, interpolated_points], dim=-1)
            else:
                new_points = torch.cat([points1.unsqueeze(-1), interpolated_points.unsqueeze(-1)], dim=-1)
                new_points = (new_points.max(dim=-1)[0] + new_points.mean(dim=-1)) / 2.0
            # new_points = interpolated_points + points1
        else:
            new_points = interpolated_points
        return new_points

    def forward(self, xyz_list, normal_list, point_list):
        xyz_list.reverse()
        point_list.reverse()
        normal_list.reverse()
        x = point_list[0]
        for i in range(self.num_stages):
            # Propagate point features to neighbors
            x = self.propagate(xyz_list[i+1], xyz_list[i], normal_list[i+1], normal_list[i], point_list[i + 1], x)

        return x
