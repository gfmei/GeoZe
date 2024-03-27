import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from libs.lib_clust import connect_clustering
from libs.lib_vis import get_colored_point_cloud_from_soft_labels
from libs.mshift import balanced_mean_shift, np_mean_shift

sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('./'))

from libs.lib_utils import (sinkhorn, index_points, down_sample, weighted_similarity, node_to_group, node_to_group_fpfh,
                            sinkhorn_rpm,
                            get_prototype, nn_assign)
from common.modules import GeoDecoder



class GAttn(nn.Module):
    def __init__(self, sigma, eps=0.01):
        super().__init__()
        self.sigma = sigma
        self.eps = eps

    def forward(self, node_gf_list, node_v_feats, weights_list, sigma_list):
        attn_score = weighted_similarity(node_gf_list, node_gf_list, weights_list, sigma_list)
        # attn_score = attn_score / attn_score.sum(dim=-1, keepdim=True)
        attn_feats = torch.matmul(attn_score, node_v_feats) + node_v_feats
        # attn_fpfhs = torch.matmul(attn_score, node_g_feats) + node_g_feats

        return attn_feats


class FPFHLAttn(nn.Module):
    def __init__(self, sigma_e=0.1, radius=0.01, n_sample=32):
        super().__init__()
        self.radius = radius
        self.n_sample = n_sample
        self.sigma_e = sigma_e

    def forward(self, down_xyz, xyz, down_fpfhs, xyz_fpfhs, xyz_feats, is_cat=False):
        # b, s, n, d
        if is_cat:
            geo_xyz = torch.cat([xyz, xyz_fpfhs], dim=-1)
            down_geo_xyz = torch.cat([down_xyz, down_fpfhs], dim=-1)
            patch_geo_xyz, patch_feats, patch_os, idx = node_to_group(
                down_geo_xyz, geo_xyz, xyz_feats, self.radius, self.n_sample, os=None, is_knn=True)
            patch_xyz = patch_geo_xyz[:, :, :, :3]
            patch_fpfhs = patch_geo_xyz[:, :, :, 3:]
        else:
            patch_xyz, patch_feats, patch_fpfhs, patch_os, idx = node_to_group_fpfh(
                down_xyz, xyz, xyz_feats, xyz_fpfhs, self.radius, self.n_sample, os=None, is_knn=True)
        v_indices = torch.cdist(patch_feats, patch_feats) / np.sqrt(patch_feats.size(-1))
        b, s, n, d = patch_feats.size()
        patch_os = patch_os.reshape(b * s, n)
        v_indices = v_indices.reshape(b * s, n, -1)
        # mask = torch.eye(n).to(xyz.device).unsqueeze(0)
        # d_indices += mask
        v_scores = sinkhorn(v_indices, patch_os, patch_os)[0].reshape(b, s, n, -1)
        g_indices = torch.cdist(patch_fpfhs, patch_fpfhs) / np.sqrt(xyz_feats.size(-1))
        g_scores = sinkhorn(g_indices.reshape(b * s, n, -1))[0].reshape(b, s, n, -1)
        attention_scores = v_scores * g_scores
        attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)
        attn_feats = torch.matmul(attention_scores, patch_feats)
        attn_feats = nn_assign(down_xyz, xyz, patch_xyz, attn_feats)
        # attn_feats = index_assign(xyz_feats, attn_feats, idx)
        # attn_fpfhs = torch.matmul(attention_scores, patch_fpfhs)
        # attn_fpfhs = index_assign(xyz_fpfhs, attn_fpfhs, idx)
        attn_feats = (xyz_feats + attn_feats) / 2.0
        return attn_feats, xyz_fpfhs


class FPFHGAttn2(nn.Module):
    def __init__(self, sigma_e=0.01, eps=0.01):
        super().__init__()
        self.sigma_e = sigma_e
        self.eps = eps

    def forward(self, node_fpfhs, node_feats, is_weight=['feats', 'geo']):
        bs, n, _ = node_fpfhs.shape
        # Create a [M, M] identity matrix
        eye_matrix = 1 - torch.eye(n, device=node_fpfhs.device)
        # Subtract the identity matrix from the tensor to set the diagonal to zero
        attn_score = 1.0
        if 'geo' in is_weight:
            n_node_fpfhs = F.normalize(node_fpfhs, dim=-1)
            d_score = torch.einsum('bmd,bnd->bmn', n_node_fpfhs, n_node_fpfhs).clip(min=0.001)
            # d_score = (d_score * eye_matrix.unsqueeze(0))
            d_score = sinkhorn_rpm(d_score / 0.01, slack=False)
            d_score = torch.exp(d_score)
            attn_score *= d_score
        if 'feats' in is_weight:
            n_node_feats = F.normalize(node_feats, dim=-1)
            f_score = torch.einsum('bmd,bnd->bmn', n_node_feats, n_node_feats).clip(min=0.01, max=0.9)
            f_score = (f_score * eye_matrix.unsqueeze(0)) + 1.0 / (n * n + 1e-6)
            f_score = sinkhorn_rpm(f_score / 0.1, slack=False)
            f_score = torch.exp(f_score)
            attn_score *= f_score
        attn_score = attn_score / attn_score.sum(dim=-1, keepdim=True)
        attn_feats = torch.matmul(attn_score, node_feats) + node_feats
        attn_fpfhs = torch.matmul(attn_score, node_fpfhs) + node_fpfhs

        return attn_feats, attn_fpfhs


class S2PAttn(nn.Module):
    def __init__(self, sigma_e=0.1, top_k=20):
        super().__init__()
        self.top_k = top_k
        self.sigma_e = sigma_e

    def forward(self, node_xyz, xyz, node_feats, xyz_feats, node_fpfhs, xyz_fpfhs, geo='gf'):
        if geo == 'dg':
            d_dis = torch.cdist(xyz, node_xyz)
            d_score = sinkhorn(d_dis)[0]
            g_dis = torch.cdist(xyz_fpfhs, node_fpfhs) / np.sqrt(node_fpfhs.size(-1))
            # g_score = torch.softmax(g_score, dim=-1) * torch.softmax(g_score, dim=-2)
            g_score = sinkhorn(g_dis)[0]
            attention_scores = d_score * g_score
            attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)
        elif geo == 'df':
            d_dis = torch.cdist(xyz, node_xyz)
            d_score = sinkhorn(d_dis)[0]
            # f_dis = torch.cdist(xyz_feats, node_feats)
            # f_score = sinkhorn(f_dis)[0]
            f_score = torch.matmul(xyz_feats, node_feats.transpose(1, 2)) / np.sqrt(node_feats.size(-1))
            # f_score = torch.exp(sinkhorn_rpm(f_score / self.sigma_e))
            f_score = torch.softmax(f_score / self.sigma_e, dim=-1) * torch.softmax(f_score / self.sigma_e, dim=-2)
            attention_scores = d_score * f_score
            attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)
        else:
            g_score = torch.matmul(xyz_fpfhs, node_fpfhs.transpose(1, 2)) / np.sqrt(node_fpfhs.size(-1))
            g_score = torch.exp(sinkhorn_rpm(g_score))
            f_score = torch.matmul(xyz_feats, node_feats.transpose(1, 2)) / np.sqrt(node_feats.size(-1))
            f_score = torch.exp(sinkhorn_rpm(f_score))
            attention_scores = g_score * f_score
            attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)
        if self.top_k > 0:
            attn_feats = get_prototype(node_feats, attention_scores.transpose(-1, -2), self.top_k)[0] + xyz_feats
            attn_fpfhs = get_prototype(node_fpfhs, attention_scores.transpose(-1, -2), self.top_k)[0] + xyz_fpfhs
        else:
            attn_feats = torch.einsum('bmn,bnd->bmd', attention_scores, node_feats) + xyz_feats
            attn_fpfhs = torch.einsum('bmn,bnd->bmd', attention_scores, node_fpfhs) + xyz_fpfhs

        return attn_feats / 2.0, attn_fpfhs / 2.0


class P2SAttn(nn.Module):
    def __init__(self, sigma_e=0.1, top_k=20):
        super().__init__()
        self.top_k = top_k
        self.sigma_e = sigma_e

    def forward(self, node_xyz, xyz, node_feats, xyz_feats, node_fpfhs, xyz_fpfhs, geo='gf'):
        if geo == 'dg':
            d_dis = torch.cdist(node_xyz, xyz) / np.sqrt(node_xyz.size(-1))
            g_dis = torch.cdist(node_fpfhs, xyz_fpfhs) / np.sqrt(node_fpfhs.size(-1))
            # g_score = torch.softmax(g_score, dim=-1) * torch.softmax(g_score, dim=-2)
            attention_scores = sinkhorn(g_dis + d_dis)[0]
            attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)
        elif geo == 'df':
            d_dis = torch.cdist(node_xyz, xyz)
            d_score = sinkhorn(d_dis)[0]
            f_score = torch.matmul(node_feats, xyz_feats.transpose(1, 2)) / np.sqrt(node_feats.size(-1))
            # f_score = torch.exp(sinkhorn_rpm(f_score / self.sigma_e))
            f_score = torch.softmax(f_score / self.sigma_e, dim=-1) * torch.softmax(f_score / self.sigma_e, dim=-2)
            attention_scores = d_score * f_score
            attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)
        else:
            g_score = torch.matmul(node_fpfhs, xyz_fpfhs.transpose(1, 2)) / np.sqrt(node_fpfhs.size(-1))
            g_score = torch.exp(sinkhorn_rpm(g_score))
            f_score = torch.matmul(node_feats, xyz_feats.transpose(1, 2)) / np.sqrt(node_feats.size(-1))
            f_score = torch.exp(sinkhorn_rpm(f_score))
            attention_scores = g_score * f_score
            attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)
        attn_feats = get_prototype(xyz_feats, attention_scores.transpose(-1, -2), self.top_k)[0] + node_feats
        attn_fpfhs = get_prototype(xyz_fpfhs, attention_scores.transpose(-1, -2), self.top_k)[0] + node_fpfhs
        attn_protos = get_prototype(xyz, attention_scores.transpose(-1, -2), self.top_k)[0] + node_xyz

        return attn_feats / 2.0, attn_fpfhs / 2.0, attn_protos / 2.0


class LAttn(nn.Module):
    def __init__(self, sigma, eps=0.01):
        super().__init__()
        self.sigma = sigma
        self.eps = eps

    def forward(self, node_gf_list, node_v_feats, weights_list, sigma_list):
        attn_score = weighted_similarity(node_gf_list, node_gf_list, weights_list, sigma_list)
        # attn_score = attn_score / attn_score.sum(dim=-1, keepdim=True)
        attn_feats = torch.matmul(attn_score, node_v_feats) + node_v_feats
        # attn_fpfhs = torch.matmul(attn_score, node_g_feats) + node_g_feats

        return attn_feats / 2.0


def g2lfusion(node_feats, feats, gamma):
    attn_feats = torch.matmul(gamma, node_feats) + feats

    return attn_feats / 2


class PartGeoZe(nn.Module):
    def __init__(self, sigma_d, sigma_a, sigma_e, angle_k, n_pts=128, voxel_size=0.05):
        super().__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.sigma_e = sigma_e
        self.angle_k = angle_k
        self.n_pts = n_pts
        self.voxel_size = voxel_size
        self.gattn = FPFHGAttn2(0.001, 32)
        self.sattn = FPFHLAttn(sigma_e, radius=0.15, n_sample=48)
        self.s2pattn = S2PAttn(0.1, 0)
        self.dec_np = GeoDecoder(1, 8)

    def forward(self, xyz, normals, fpfhs, feats, text_feats):
        # new_feats = copy.deepcopy(feats).view(-1, feats.size(-1)).cpu().detach().numpy()
        # new_fpfhs = copy.deepcopy(fpfhs).view(-1, fpfhs.size(-1)).cpu().detach().numpy()
        # v_cents, g_cents, _ = balanced_mean_shift(new_feats, new_fpfhs, 4, 1.0, eps=0.5, iters=10, alpha=3)
        # v_cents, g_cents, labels_unique = np_mean_shift(new_feats, new_fpfhs)
        # v_cents = v_cents.to(feats)
        # g_cents = g_cents.to(fpfhs)
        # print(soft_labels.shape)
        # get_colored_point_cloud_from_labels(xyz[0].cpu().numpy(), soft_labels[0].cpu().numpy(), 'text')
        node_xyz, node_feats, idx = down_sample(xyz, feats, self.n_pts)
        node_normals = index_points(normals, idx)
        protos = node_xyz
        feat_list = [normals, fpfhs]
        node_fpfhs = index_points(fpfhs, idx)
        w_list = [1.0, 0.1]
        d_type = ['cos', 'eu']
        # eig_values = calculate_curvature_pca_ball(xyz, node_xyz, num_neighbors=10, radius=0.1, eps=1e-8)
        cts_list = [node_normals, node_fpfhs]
        gamma, protos, cts_list = connect_clustering(xyz, protos, feat_list, cts_list, w_list, d_type, iters=30)
        # gamma, pi, cts_list = fusion_wkeans(feat_list, w_list, cts_list, d_type, iters=30, is_prob=False, idx=0)
        node_normals = cts_list[0]
        # # node_feats = gmm_params(gamma, feats)[1]
        # # feats = g2lfusion(node_feats, feats, gamma)
        top_k = xyz.size(1) // self.n_pts
        node_fpfhs = get_prototype(fpfhs, gamma, top_k)[0]
        protos = get_prototype(xyz, gamma, top_k)[0]
        feats, _ = self.sattn(protos, xyz, node_fpfhs, fpfhs, feats, True)
        node_feats = get_prototype(feats, gamma, top_k)[0]
        node_feats, node_fpfhs = self.gattn(node_fpfhs, node_feats)
        feats, fpfhs = self.s2pattn(protos, xyz, node_feats, feats, node_fpfhs, fpfhs, 'df')
        feats = self.dec_np([xyz, protos], [normals, node_normals], [feats, node_feats])
        # v_cents, g_cents, _ = balanced_mean_shift(feats, fpfhs, 4, 1.0, eps=0.5, iters=10, alpha=3)
        # soft_labels = torch.sigmoid(torch.einsum('bmd,nd->bmn', feats, v_cents)) * torch.sigmoid(
        #     torch.einsum('bmd,nd->bmn', fpfhs, g_cents))
        # soft_ids = torch.argmax(soft_labels, dim=-1).unsqueeze(-1).expand_as(feats)

        return feats, node_feats, idx
