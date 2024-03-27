import torch
from torch import nn
import torch.nn.functional as F
from torch_cluster import knn_graph
import numpy as np

from libs.lib_clust import get_spt_centers
from libs.lib_utils import sinkhorn


def get_spt_graph(points, feats, spt_ids, k=8):
    assert points.dim() == 2
    # ids = torch.unique(spt_ids, return_inverse=True)[1]
    # print(ids)
    featcat = torch.cat([points, feats], dim=-1)
    super_center, uni_ids = get_spt_centers(featcat, spt_ids)
    spt_points = super_center[:, :3]
    spt_feats = super_center[:, 3:]
    spt_feats = F.normalize(spt_feats, dim=-1)
    edges = knn_graph(spt_points, k=k).T

    return spt_points, spt_feats, uni_ids, edges


class GAttn(nn.Module):
    pass


class LAttn(nn.Module):
    def __init__(self):
        super(LAttn, self).__init__()

    def forward(self, v_feats, g_feats, spt_feats, uni_ids, ids):
        feats = torch.zeros_like(v_feats)
        for idx in uni_ids:
            mask = np.where(ids == idx)[0]
            v_feat_i = F.normalize(v_feats[mask], dim=-1)
            # g_feat_i = F.normalize(g_feats[mask], dim=-1)
            # g_feat_i = g_feats[mask]
            print(v_feat_i.shape)
            v_indices = 1 - torch.einsum('md,nd->mn', v_feat_i, v_feat_i).unsqueeze(0)
            # feats[np.where(ids == idx)[0]] = spt_feats[idx]
            v_scores = sinkhorn(v_indices)[0].squeeze(0)
            # g_indices = 1 - torch.einsum('md,nd->mn', g_feat_i, g_feat_i).unsqueeze(0)
            # g_indices = torch.cdist(g_feat_i, g_feat_i).unsqueeze(0)
            # g_scores = sinkhorn(g_indices)[0].squeeze(0)
            attention_scores = v_scores
            attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)
            feats[mask] = torch.matmul(attention_scores.half(), v_feats[mask])
        return feats


class SemGeoze(nn.Module):
    def __init__(self, dim=768, sigma_d=0.01, sigma_a=0.001, sigma_e=0.001, angle_k=10,
                 voxel_size=0.05, n_pts=2000, lk=8):
        super().__init__()

        self.lattn = LAttn()
        self.lk = lk

    def forward(self, xyz, normals, fpfhs, feats, spt_ids):
        spt_ids = np.asarray(spt_ids)
        spt_points, spt_feats, uni_ids, edges = get_spt_graph(xyz, feats, spt_ids, k=self.lk)
        new_feats = self.lattn(feats, xyz, spt_feats, uni_ids, spt_ids)
        return new_feats
