import copy

import numpy as np
import torch
import torch.nn.functional as F

try:
    import segmentator
    from torch_cluster import knn_graph
except Exception as e:
    pass

from libs.lib_utils import farthest_point_sample, index_points, sinkhorn, angle_difference, get_prototype


def graph_cut_clustering(points, normals, thres=0.01, k=50, seg_min_verts=20):
    if points.dim() == 2:
        points = points.unsqueeze(0)
        normals = normals.unsqueeze(0)
    superpoints = list()
    for point, normal in zip(points, normals):
        edges = knn_graph(point, k=k).T
        index = segmentator.segment_point(point, normal, edges, kThresh=thres, segMinVerts=seg_min_verts)
        superpoints.append(index)
    return torch.cat(superpoints, dim=0)


def pairwise_histogram_distance_pytorch(hist1, hist2):
    """
    Calculate the pairwise histogram intersection distance between batches of point clouds A and B,
    optimized for PyTorch with support for batch processing.

    Parameters:
    - histograms_A: Histograms for each point in point clouds A, shape: [B, N, D].
    - histograms_B: Histograms for each point in point clouds B, shape: [B, M, D].

    Returns:
    - distance_matrix: A tensor of distances where element (b, i, j) is the distance from the i-th point in the b-th point cloud in A to the j-th point in the b-th point cloud in B.
    """
    # Ensure input tensors are float for division
    hist1 = hist1.float()
    hist2 = hist2.float()

    # Expand histograms for broadcasting: [B, N, 1, D] and [B, 1, M, D]
    hist1_exp = hist1.unsqueeze(2)
    hist2_exp = hist2.unsqueeze(1)

    # Calculate minimum of each pair of histograms, resulting in a [B, N, M, D] tensor
    minima = torch.min(hist1_exp, hist2_exp)

    # Sum over the last dimension (D) to get the intersection values, resulting in a [B, N, M] tensor
    intersections = torch.sum(minima, dim=-1)

    # Normalize the intersections to get a similarity measure and then convert to distances
    sum1 = torch.sum(hist1, dim=-1, keepdim=True)  # Shape [B, N, 1]
    sum2 = torch.sum(hist2, dim=-1, keepdim=True)  # Shape [B, 1, M]
    max_sum = torch.min(sum1, sum2.transpose(1, 2))  # Shape [B, N, M]

    normalized_intersections = intersections / max_sum
    distance_matrix = 1 - normalized_intersections

    return distance_matrix


def pairwise_histogram_distance_optimized(hist1, hist2):
    """
    Calculate the pairwise histogram intersection distance between each point in A to every point in B,
    optimized to reduce explicit for loops.

    Parameters:
    - histograms_A: Histograms for each point in point cloud A (shape: [N, D]).
    - histograms_B: Histograms for each point in point cloud B (shape: [M, D]).

    Returns:
    - distance_matrix: A matrix of distances where element (i, j) is the distance from the i-th point in A to the j-th point in B.
    """
    # Expand histograms_A and histograms_B to 3D tensors for broadcasting
    # histograms_A: [N, 1, D], histograms_B: [1, M, D]
    hist1_exp = hist1[:, np.newaxis, :]
    hist2_exp = hist2[np.newaxis, :, :]

    # Calculate minimum of each pair of histograms (using broadcasting), resulting in a [N, M, D] tensor
    minima = np.minimum(hist1_exp, hist2_exp)

    # Sum over the last dimension (D) to get the intersection values, resulting in a [N, M] matrix
    intersections = np.sum(minima, axis=2)

    # Calculate normalized intersections as distances
    sum1 = np.sum(hist1, axis=1)[:, np.newaxis]  # Shape [N, 1]
    sum2 = np.sum(hist2, axis=1)[np.newaxis, :]  # Shape [1, M]
    max_sum = np.minimum(sum1, sum2)  # Broadcasting to get max sum for each pair

    normalized_intersections = intersections / max_sum
    distance_matrix = 1 - normalized_intersections

    return distance_matrix


def gmm_params(gamma, pts, return_sigma=False):
    """
    gamma: B feats N feats J
    pts: B feats N feats D
    """
    # pi: B feats J
    D = pts.size(-1)
    pi = gamma.mean(dim=1)
    npi = pi * gamma.shape[1] + 1e-5
    # p: B feats J feats D
    mu = gamma.transpose(1, 2) @ pts / npi.float().unsqueeze(2)
    if return_sigma:
        # diff: B feats N feats J feats D
        diff = pts.unsqueeze(2) - mu.unsqueeze(1)
        # sigma: B feats J feats 3 feats 3
        eye = torch.eye(D).unsqueeze(0).unsqueeze(1).to(gamma.device)
        sigma = (((diff.unsqueeze(3) @ diff.unsqueeze(4)).squeeze() *
                  gamma).sum(dim=1) / npi).unsqueeze(2).unsqueeze(3) * eye
        return pi, mu, sigma
    return pi, mu


def clu_dis(pts, mu, d_type='eu'):
    if d_type == 'ang':
        return angle_difference(pts, mu)
    elif d_type == 'hist':
        return pairwise_histogram_distance_pytorch(pts, mu)
    elif d_type == 'cos':
        return torch.cdist(F.normalize(pts, dim=-1), F.normalize(mu, dim=-1))
    else:
        dis = torch.cdist(pts, mu)
        # min_d = torch.min(dis, dim=-1)[0].min(dim=-1)[0]
        # max_d = torch.max(dis, dim=-1)[0].max(dim=-1)[0]
        # gap = (max_d - min_d).clip(min=1e-4)
        # dis = (dis - min_d.view(-1, 1, 1)) / gap.view(-1, 1, 1)
        return dis


def norm_mu(mu, d_type='eu'):
    if d_type in ['ang', 'cos']:
        return F.normalize(mu, dim=-1)
    else:
        return mu


def connect_clustering(xyz, mu_xyz, feat_list, cts_list, w_list, d_type, iters=30, idx=None):
    bs, n, _ = xyz.shape
    n_clus = mu_xyz.shape[1]
    device = mu_xyz.device
    dist_xyz = torch.cdist(xyz, mu_xyz)
    eps = torch.topk(dist_xyz, k=2, largest=False, dim=-1)[0][:, :, 1].mean(dim=-1).view(bs, 1, 1)
    gamma, pi = torch.ones((bs, n, n_clus), device=device), None
    for _ in range(iters):
        cost_list = list()
        for i in range(len(feat_list)):
            dist_i = w_list[i] * clu_dis(feat_list[i], cts_list[i], d_type[i])
            cost_list.append(dist_i)
        dist_xyz = torch.cdist(xyz, mu_xyz)
        w_xyz = np.sqrt(1 / 3) / eps.clip(min=1e-4).expand(bs, n, n_clus)
        cost_list.append(w_xyz * dist_xyz)
        cost = torch.stack(cost_list, dim=0)
        cost = cost.mean(dim=0)
        if idx is not None:
            gamma = sinkhorn(cost, q=pi, max_iter=10)[0]
            pi = gamma.mean(dim=1)
        else:
            gamma = sinkhorn(cost, q=None, max_iter=10)[0]
        # print(gamma[0].sum(), 'label0')
        # dist_xyz = torch.cdist(xyz, mu_xyz)
        mask = (dist_xyz < (np.sqrt(5) - 1) / 2.0 * eps).to(xyz)
        mask_prob = (gamma * n * n_clus > 1).to(gamma)
        m_gamma = mask * gamma * mask_prob
        # print(gamma[0].sum(), 'label1')
        m_gamma = m_gamma / m_gamma.sum(dim=-1, keepdim=True).clip(min=1e-3)
        cts_list = [norm_mu(gmm_params(m_gamma, feat)[1], dt) for feat, dt in zip(feat_list, d_type)]
        # mu_xyz = gmm_params(m_gamma, xyz)[1]
        # eps = torch.topk(dist_xyz, k=2, largest=False, dim=-1)[0][:, :, 1].mean(dim=-1).view(bs, 1, 1)
        # mask = (dist_xyz < np.sqrt(3) * eps).to(xyz)

    return gamma, mu_xyz, cts_list


def norm_cost(cost):
    cost_min = cost.min(dim=-1)[0].min(dim=-1)[0]
    cost_max = cost.max(dim=-1)[0].max(dim=-1)[0]
    in_gap_dis = 1.0 / (cost_max - cost_min).clip(min=1e-3)
    cost = torch.einsum('bmn,b->bmn', cost, in_gap_dis)

    return cost


def wkeans_plus(x, centroids, iters=10, top_k=20, is_norm=False, is_xyz=False, tau=0.1, is_prob=False):
    bs, num, dim = x[0].shape
    device = x[0].device
    num_clusters = centroids[0].size(-2)
    gamma, pi = torch.ones((bs, num, num_clusters), device=device), None
    for i in range(iters):
        cost = torch.zeros((bs, num, num_clusters), device=device)
        for k in range(len(x)):
            cnt_xyz = centroids[k]
            xyz = x[k]
            cost_xyz = torch.cdist(xyz, cnt_xyz).clip(min=0.0)
            if len(x) > 1:
                cost_xyz = norm_cost(cost_xyz) / tau
            cost += cost_xyz
        if is_prob:
            gamma = sinkhorn(cost, q=pi, max_iter=10)[0]
        else:
            gamma = sinkhorn(cost, q=None, max_iter=10)[0]
        gamma = gamma / gamma.sum(dim=-1, keepdim=True).clip(min=1e-3)
        if top_k > 0:
            for k in range(len(centroids)):
                centroids[k], _ = get_prototype(x[k], gamma, top_k=top_k)
        else:
            for k in range(len(centroids)):
                _, centroids[k] = gmm_params(gamma, x[k])
        if is_prob:
            prob = torch.softmax(-torch.cdist(x[-1], centroids[-1] / tau), dim=-1) * gamma
            prob = prob.sum(1)
            pi = prob / prob.sum(-1, keepdim=True).clip(min=1e-3)
        if is_norm:
            if is_xyz:
                start = 1
            else:
                start = 0
            for k in range(start, len(centroids)):
                centroids[k] = F.normalize(centroids[k], dim=-1)

    return gamma, pi, centroids


def clustering(xyz, feats, n_clus, idx=None, iters=50, is_prob=False, tau=0.1):
    if idx is None:
        idx = farthest_point_sample(xyz, n_clus)
    new_xyz = index_points(xyz, idx)
    if feats is None:
        score, pi, cents = wkeans_plus([xyz], [new_xyz], iters=iters, top_k=-1, is_norm=True,
                                       is_xyz=True, is_prob=is_prob, tau=tau)
    else:
        new_feats = [index_points(feat, idx) for feat in feats]
        score, pi, cents = wkeans_plus([xyz] + feats, [new_xyz] + new_feats, iters=iters, top_k=-1,
                                       is_norm=True, is_xyz=True, is_prob=is_prob, tau=tau)

    return score, pi, cents, idx


def get_spt_centers(points, ids):
    uni_ids = np.unique(ids)
    spnum = len(uni_ids)
    superpoint_center = torch.zeros((spnum, points.size(-1))).to(points)
    for spID in uni_ids:
        spMask = np.where(ids == spID)[0]
        try:
            superpoint_center[spID] = points[spMask].mean(dim=0)
        except IndexError:
            print(spID, 'not found', uni_ids.max(), spnum)

    return superpoint_center, uni_ids


if __name__ == '__main__':
    pts = torch.rand(3, 1024, 32)
