import copy

import numpy as np
import torch
from torch import nn
from transformers.models.esm.openfold_utils import feats


def angle_difference(src_feats, dst_feats):
    """Calculate angle between each pair of vectors.
    Assumes points are l2-normalized to unit length.

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src_feats.shape
    _, M, _ = dst_feats.shape
    dist = torch.matmul(src_feats, dst_feats.permute(0, 2, 1))
    dist = torch.acos(dist)

    return dist


def index_points(points, idx):
    """Array indexing, i.e. retrieves relevant points based on indices

    Args:
        points: input points data_loader, [B, N, C]
        idx: sample index data_loader, [B, S]. S can be 2 dimensional
    Returns:
        new_points:, indexed points data_loader, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, n_point, is_center=False):
    """
    Input:
        pts: point cloud data, [B, N, 3]
        n_point: number of samples
    Return:
        sub_xyz: sampled point cloud index, [B, n_point]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, n_point, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(xyz) * 1e10
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    if is_center:
        centroid = xyz.mean(1).view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    for i in range(n_point):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def angle(v1: torch.Tensor, v2: torch.Tensor):
    """Compute angle between 2 vectors

    For robustness, we use the same formulation as in PPFNet, i.e.
        angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
    This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0

    Args:
        v1: (B, *, 3)
        v2: (B, *, 3)

    Returns:

    """
    # Normalize v1 and v2 to get unit vectors
    v1_norm = v1 / v1.norm(dim=-1, keepdim=True)
    v2_norm = v2 / v2.norm(dim=-1, keepdim=True)

    # Compute dot product between each pair in v1 and v2
    # Since v1 and v2 might have different lengths (N != M), we cannot directly compute the dot product across batches
    # We need to manually broadcast them to align their dimensions

    # Expand v1 and v2 to make their shapes compatible for broadcasting
    v1_expanded = v1_norm.unsqueeze(2)  # Shape becomes [B, N, 1, 3]
    v2_expanded = v2_norm.unsqueeze(1)  # Shape becomes [B, 1, M, 3]

    # Compute dot products, result shape will be [B, N, M]
    dot_products = torch.sum(v1_expanded * v2_expanded, dim=-1)

    # Ensure the dot product values are within the range [-1, 1] to avoid NaN errors due to floating point arithmetic issues
    dot_products = torch.clamp(dot_products, -1.0, 1.0)

    # Compute angles using arccosine, result shape will be [B, N, M]
    angles = torch.acos(dot_products)

    return angles


def normal_redirect(points, normals, view_point):
    '''
    Make direction of normals towards the view point
    '''
    vec_dot = np.sum((view_point - points) * normals, axis=-1)
    mask = (vec_dot < 0.)
    redirected_normals = normals.copy()
    redirected_normals[mask] *= -1.
    return redirected_normals


def calculate_curvature_pca(points, neighbors, eps=1e-8):
    """
    :param points: B, N, C
    :param neighbors: N, N, S, C
    :param eps:
    :return:
    """
    # Calculate covariance matrices
    num_neighbors = neighbors.shape[-2]
    centered_neighbor_points = neighbors - points.unsqueeze(-2)
    cov_matrices = torch.matmul(centered_neighbor_points.transpose(-2, -1), centered_neighbor_points) / num_neighbors

    # Calculate eigenvalues and curvatures
    eigenvalues = torch.linalg.eigvalsh(cov_matrices)
    max_eigenvalues = eigenvalues.max(dim=-1)[0]
    curvatures = 2 * max_eigenvalues / (eigenvalues.sum(dim=-1) + eps)

    return curvatures


def index_gather(points, idx):
    """
    Input:
        points: input feats semdata, [B, N, C]
        idx: sample index semdata, [B, S, K]
    Return:
        new_points:, indexed feats semdata, [B, S, K, C]
    """
    dim = points.size(-1)
    n_clu = idx.size(1)
    # device = points.device
    view_list = list(idx.shape)
    view_len = len(view_list)
    # feats_shape = view_list
    xyz_shape = [-1] * (view_len + 1)
    xyz_shape[-1] = dim
    feats_shape = [-1] * (view_len + 1)
    feats_shape[1] = n_clu
    batch_indices = idx.unsqueeze(-1).expand(xyz_shape)
    points = points.unsqueeze(1).expand(feats_shape)
    new_points = torch.gather(points, dim=-2, index=batch_indices)
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz, itself_indices=None):
    """ Grouping layer in PointNet++.

    Inputs:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, (B, N, C)
        new_xyz: query points, (B, S, C)
        itself_indices (Optional): Indices of new_xyz into xyz (B, S).
          Used to try and prevent grouping the point itself into the neighborhood.
          If there is insufficient points in the neighborhood, or if left is none, the resulting cluster will
          still contain the center point.
    Returns:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # (B, S, N)
    sqrdists = torch.cdist(new_xyz, xyz)

    if itself_indices is not None:
        # Remove indices of the center points so that it will not be chosen
        batch_indices = torch.arange(B, dtype=torch.long).to(device)[:, None].repeat(1, S)  # (B, S)
        row_indices = torch.arange(S, dtype=torch.long).to(device)[None, :].repeat(B, 1)  # (B, S)
        group_idx[batch_indices, row_indices, itself_indices] = N

    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    if itself_indices is not None:
        group_first = itself_indices[:, :, None].repeat([1, 1, nsample])
    else:
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx.clip(min=0, max=N)


def calculate_curvature_pca_ball(queries, refs, num_neighbors=10, radius=0.1, eps=1e-8):
    # batch_size, num_points, num_features = points.size()
    idx = query_ball_point(radius, num_neighbors, refs, queries)  # [B, K, n_sampling]
    mean_node = torch.mean(queries, dim=-2, keepdim=True)
    cat_points = torch.cat([refs, mean_node], dim=1)
    os = torch.ones((refs.shape[0], refs.shape[1])).to(refs)
    neighbor_points = index_gather(cat_points, idx)  # [B, n_point, n_sample, 3]
    cat_os = torch.cat([os, torch.zeros_like(os[:, :1])], dim=-1).unsqueeze(-1)
    neighbor_os = index_gather(cat_os, idx).squeeze(-1)
    # Calculate covariance matrices
    inners = torch.sum(neighbor_os, dim=-1, keepdim=True)
    # w_neighbor_points = torch.einsum('bnkd,bnk->bnkd', neighbor_points, neighbor_os) / inners.unsqueeze(-1)
    centered_neighbor_points = neighbor_points - queries.unsqueeze(2)
    w_centered_neighbor_points = torch.einsum(
        'bnkd,bnk->bnkd', centered_neighbor_points, neighbor_os) / inners.unsqueeze(-1)
    cov_matrices = torch.matmul(centered_neighbor_points.transpose(-2, -1), w_centered_neighbor_points)
    # Calculate eigenvalues and curvatures
    eigenvalues = torch.linalg.eigvalsh(cov_matrices + eps)
    lmd = [eigenvalues[:, :, 2].clip(min=10*eps), eigenvalues[:, :, 1], eigenvalues[:, :, 0]]
    features = [(lmd[0] - lmd[1]) / lmd[0], (lmd[1] - lmd[2]) / lmd[0],  lmd[2] / lmd[0]]

    return torch.stack(features, dim=-1)


def log_boltzmann_kernel(cost, u, v, epsilon):
    kernel = (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon
    return kernel


def sinkhorn(cost, p=None, q=None, epsilon=1e-4, thresh=1e-2, max_iter=100):
    if p is None or q is None:
        batch_size, num_x, num_y = cost.shape
        device = cost.device
        if p is None:
            p = torch.empty(batch_size, num_x, dtype=torch.float,
                            requires_grad=False, device=device).fill_(1.0 / num_x).squeeze()
        if q is None:
            q = torch.empty(batch_size, num_y, dtype=torch.float,
                            requires_grad=False, device=device).fill_(1.0 / num_y).squeeze()
    # Initialise approximation vectors in log domain
    u = torch.zeros_like(p).to(p)
    v = torch.zeros_like(q).to(q)
    # Stopping criterion, sinkhorn iterations
    for i in range(max_iter):
        u0, v0 = u, v
        # u^{l+1} = a / (K v^l)
        K = log_boltzmann_kernel(cost, u, v, epsilon)
        u_ = torch.log(p + 1e-8) - torch.logsumexp(K, dim=-1)
        u = epsilon * u_ + u
        # v^{l+1} = b / (K^T u^(l+1))
        Kt = log_boltzmann_kernel(cost, u, v, epsilon).transpose(-2, -1)
        v_ = torch.log(q + 1e-8) - torch.logsumexp(Kt, dim=-1)
        v = epsilon * v_ + v
        # Size of the change we have performed on u
        diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
        mean_diff = torch.mean(diff)
        if mean_diff < thresh:
            break
    # Transport plan pi = diag(a)*K*diag(b)
    K = log_boltzmann_kernel(cost, u, v, epsilon)
    gamma = torch.exp(K)
    # Sinkhorn distance
    return gamma, K


def down_sample(points, features, n_point):
    idx = farthest_point_sample(points, n_point, True)
    down_pts = index_points(points, idx)
    down_fts = index_points(features, idx)
    return down_pts, down_fts, idx


def index_assign(feats, patch_feats, idx):
    bs, N, dim = feats.shape
    feats_copy = copy.deepcopy(feats)
    patch_feats = patch_feats.reshape(bs, -1, dim)
    idx = idx.reshape(bs, -1)
    batch_idx = torch.arange(bs)[:, None].to(idx.device).expand_as(idx)
    feats_copy[batch_idx, idx] = patch_feats
    return feats_copy


def sinkhorn_rpm(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)
            log_alpha_padded = torch.nan_to_num(log_alpha_padded, nan=0.0)
            # Column normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)
            log_alpha_padded = torch.nan_to_num(log_alpha_padded, nan=0.0)
            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))
            log_alpha = torch.nan_to_num(log_alpha, nan=0.0)
            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))
            log_alpha = torch.nan_to_num(log_alpha, nan=0.0)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha


def weighted_similarity(src_list, tgt_list, weights_list, sigma_list):
    sim_list = [weight * sinkhorn(torch.cdist(src, tgt) / sigma)[1]
                for src, tgt, weight, sigma in zip(src_list, tgt_list, weights_list, sigma_list)]
    scores = torch.exp(torch.stack(sim_list, dim=0).mean(dim=0))
    scores = scores / scores.sum(dim=-1, keepdim=True).clip(min=1e-4)

    return scores


def knn_query_group(nsample, xyz, new_xyz):
    """
    Input:
        n_sample: max sample number in local region
        xyz: all feats, [B, N, 3]
        new_xyz: query feats, [B, S, 3]
    Return:
        group_idx: grouped feats index, [B, S, n_sample]
    """
    if not isinstance(xyz, torch.Tensor):
        xyz = torch.from_numpy(xyz)
    if not isinstance(new_xyz, torch.Tensor):
        new_xyz = torch.from_numpy(xyz)
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    xyz_, new_xyz_ = xyz, new_xyz
    if C > 3:
        xyz_[:, :, 3:] = xyz[:, :, 3:] / np.sqrt(C - 3)
        xyz_[:, :, :3] = xyz[:, :, :3] / np.sqrt(3)
        new_xyz_[:, :, 3:] = new_xyz[:, :, 3:] / np.sqrt(C - 3)
        new_xyz_[:, :, :3] = new_xyz[:, :, :3] / np.sqrt(3)
    sqrdists = torch.cdist(new_xyz_, xyz_)
    idx = torch.topk(sqrdists, nsample + 1, largest=False, dim=-1)[1]  # [B, S, nsample+1]
    group_idx = idx.clone()[:, :, :nsample]

    return group_idx.clip(min=0, max=N)


def node_to_group(node, xyz, feats, radius, n_sample, os=None, is_knn=True):
    """
    Input:
        os: [B, N]
        radius: ball radius
        n_sample:
        xyz: input feats position semdata, [B, N, 3]
        feats: input feats semdata, [B, N, D]
    Return:
        new_xyz: sampled feats position semdata, [B, n_point, n_sample, 3]
        new_points: sampled feats semdata, [B, n_point, n_sample, D]
    """
    if os is None:
        xyz_center = torch.mean(xyz, dim=1, keepdim=True)
        feats_center = torch.mean(feats, dim=1, keepdim=True)
        os = torch.ones((xyz.shape[0], xyz.shape[1])).to(xyz) / xyz.shape[1]
    else:
        if len(os.shape) == 1:
            os = os.unsqueeze(0)
        xyz_center = torch.einsum('bnd,bn->bd', xyz, os)
        pi = torch.sum(os, dim=-1, keepdim=True).clip(min=1e-4)
        xyz_center = (xyz_center / pi).unsqueeze(1)
        feats_center = torch.einsum('bnd,bn->bd', feats, os)
        feats_center = (feats_center / pi).unsqueeze(1)
    if is_knn:
        idx = knn_query_group(n_sample, xyz, node)
    else:
        idx = query_ball_point(radius, n_sample, xyz, node)  # [B, K, n_sampling]
    cat_xyz = torch.cat([xyz, xyz_center], dim=1)
    grouped_xyz = index_gather(cat_xyz, idx)  # [B, n_point, n_sample, 3]
    cat_feats = torch.cat([feats, feats_center], dim=1)
    grouped_feats = index_gather(cat_feats, idx)  # [B, n_point, n_sample, C]
    cat_os = torch.cat([os, torch.zeros_like(os[:, :1])], dim=-1).unsqueeze(-1)
    grouped_os = index_gather(cat_os, idx)
    return grouped_xyz, grouped_feats, grouped_os.squeeze(-1), idx



def node_to_group(node, xyz, feats, radius, n_sample, os=None, is_knn=True):
    """
    Input:
        os: [B, N]
        radius: ball radius
        n_sample:
        xyz: input feats position semdata, [B, N, 3]
        feats: input feats semdata, [B, N, D]
    Return:
        new_xyz: sampled feats position semdata, [B, n_point, n_sample, 3]
        new_points: sampled feats semdata, [B, n_point, n_sample, D]
    """
    if os is None:
        xyz_center = torch.mean(xyz, dim=1, keepdim=True)
        feats_center = torch.mean(feats, dim=1, keepdim=True)
        os = torch.ones((xyz.shape[0], xyz.shape[1])).to(xyz) / xyz.shape[1]
    else:
        if len(os.shape) == 1:
            os = os.unsqueeze(0)
        xyz_center = torch.einsum('bnd,bn->bd', xyz, os)
        pi = torch.sum(os, dim=-1, keepdim=True).clip(min=1e-4)
        xyz_center = (xyz_center / pi).unsqueeze(1)
        feats_center = torch.einsum('bnd,bn->bd', feats, os)
        feats_center = (feats_center / pi).unsqueeze(1)
    if is_knn:
        idx = knn_query_group(n_sample, xyz, node)
    else:
        idx = query_ball_point(radius, n_sample, xyz, node)  # [B, K, n_sampling]
    cat_xyz = torch.cat([xyz, xyz_center], dim=1)
    grouped_xyz = index_gather(cat_xyz, idx)  # [B, n_point, n_sample, 3]
    cat_feats = torch.cat([feats, feats_center], dim=1)
    grouped_feats = index_gather(cat_feats, idx)  # [B, n_point, n_sample, C]
    cat_os = torch.cat([os, torch.zeros_like(os[:, :1])], dim=-1).unsqueeze(-1)
    grouped_os = index_gather(cat_os, idx)
    return grouped_xyz, grouped_feats, grouped_os.squeeze(-1), idx


def node_to_group_fpfh(node, xyz, feats, fpfhs, radius, n_sample, os=None, is_knn=True):
    """
    Input:
        os: [B, N]
        radius: ball radius
        n_sample:
        xyz: input feats position semdata, [B, N, 3]
        feats: input feats semdata, [B, N, D]
    Return:
        new_xyz: sampled feats position semdata, [B, n_point, n_sample, 3]
        new_points: sampled feats semdata, [B, n_point, n_sample, D]
    """
    if os is None:
        xyz_center = torch.mean(xyz, dim=1, keepdim=True)
        feat_center = torch.mean(feats, dim=1, keepdim=True)
        fpfh_center = torch.mean(fpfhs, dim=1, keepdim=True)
        os = torch.ones((xyz.shape[0], xyz.shape[1])).to(xyz) / xyz.shape[1]
    else:
        if len(os.shape) == 1:
            os = os.unsqueeze(0)
        xyz_center = torch.einsum('bnd,bn->bd', xyz, os)
        pi = torch.sum(os, dim=-1, keepdim=True).clip(min=1e-4)
        xyz_center = (xyz_center / pi).unsqueeze(1)
        feat_center = torch.einsum('bnd,bn->bd', feats, os)
        feat_center = (feat_center / pi).unsqueeze(1)
        fpfh_center = torch.einsum('bnd,bn->bd', fpfhs, os)
        fpfh_center = (fpfh_center / pi).unsqueeze(1)
    if is_knn:
        idx = knn_query_group(n_sample, xyz, node)
    else:
        idx = query_ball_point(radius, n_sample, xyz, node)  # [B, K, n_sampling]
    cat_xyz = torch.cat([xyz, xyz_center], dim=1)
    grouped_xyz = index_gather(cat_xyz, idx)  # [B, n_point, n_sample, 3]
    cat_feats = torch.cat([feats, feat_center], dim=1)
    cat_fpfhs = torch.cat([fpfhs, fpfh_center], dim=1)
    grouped_feats = index_gather(cat_feats, idx)  # [B, n_point, n_sample, C]
    cat_os = torch.cat([os, torch.zeros_like(os[:, :1])], dim=-1).unsqueeze(-1)
    grouped_fpfhs = index_gather(cat_fpfhs, idx)  # [B, n_point, n_sample, C]
    grouped_os = index_gather(cat_os, idx)
    return grouped_xyz, grouped_feats, grouped_fpfhs, grouped_os, idx


def get_prototype(features, similarity, top_k=20):
    """
    :param features: [B, N, D]
    :param similarity: [B, N, K]
    :return:
    """
    similarity = similarity.transpose(-1, -2)  # b, k, n
    vals, ids = torch.topk(similarity, top_k, largest=True, dim=-1)  # b, k, top_k
    ids = ids.unsqueeze(dim=-1).expand(-1, -1, -1, features.shape[-1])
    patch_features = torch.gather(features.unsqueeze(1).expand(-1, similarity.shape[1], -1, -1),
                                  dim=-2, index=ids)  # b, k, top_k, d
    vals = vals / torch.sum(vals, keepdim=True, dim=-1).clip(1e-4)
    protos = torch.einsum('bksd,bks->bkd', patch_features, vals)
    prob = vals.sum(-1)

    return protos, prob / prob.mean(dim=-1, keepdim=True).clip(min=1e-4)



def nn_assign(down_xyz, xyz, patch_xyz, patch_feats):
    idx = knn_query_group(1, down_xyz, xyz)  # [B, N, 1]
    B, S, K, D = patch_feats.size()
    expanded_idx = idx.unsqueeze(-1)
    nn_patch_xyz = torch.gather(patch_xyz, dim=1, index=expanded_idx.expand(B, -1, K, down_xyz.size(-1)))
    nn_patch_feats = torch.gather(patch_feats, dim=1, index=expanded_idx.expand(B, -1, K, D))
    nn_dis = torch.cdist(xyz.unsqueeze(-2), nn_patch_xyz)
    ids = torch.topk(nn_dis, k=1, dim=-1, largest=False)[1]  # [B, N, 1]
    return torch.gather(nn_patch_feats, index=ids.expand(B, -1, -1, D), dim=-2).squeeze(-2)


