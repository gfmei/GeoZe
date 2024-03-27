import copy

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN

from libs.lib_clust import gmm_params
from libs.lib_utils import sinkhorn


def normalize_matrix(matrix, axis=1, norm='l2'):
    """
    Normalize a NumPy matrix along a specified axis.

    Parameters:
    - matrix: A NumPy array to be normalized.
    - axis: The axis along which normalization is performed. Default is 1.
    - norm: The type of norm to use for normalization. Supports 'l1' and 'l2'. Default is 'l2'.

    Returns:
    - A normalized NumPy array.
    """
    if norm == 'l2':
        # Compute the L2 norm of each row (axis=1) or column (axis=0)
        l2_norm = np.linalg.norm(matrix, axis=axis, keepdims=True)
        # Avoid division by zero
        l2_norm[l2_norm == 0] = 1
        # Normalize the matrix
        normalized_matrix = matrix / l2_norm
    elif norm == 'l1':
        # Compute the L1 norm of each row (axis=1) or column (axis=0)
        l1_norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        # Avoid division by zero
        l1_norm[l1_norm == 0] = 1
        # Normalize the matrix
        normalized_matrix = matrix / l1_norm
    else:
        raise ValueError("Unsupported norm. Use 'l1' or 'l2'.")

    return normalized_matrix


def similarity(src, tgt):
    src_norm = F.normalize(src, dim=-1)
    tgt_norm = F.normalize(tgt, dim=-1)
    scores = torch.matmul(src_norm, tgt_norm.transpose(1, 2))
    return scores


def one_step_mean_shift(feats, centroids, mu, nu=None, tau=1.0, eps=0.5):
    scores = similarity(feats, centroids)
    cost = 1 - copy.deepcopy(scores)
    gamma = sinkhorn(cost / tau, p=mu, q=nu)[0]
    mask = (scores >= eps).to(scores)
    gamma = gamma * mask
    gamma = gamma / torch.sum(gamma, dim=-1, keepdim=True).clip(min=1e-3)
    pi, delta = gmm_params(gamma, feats)
    # selected_ids = torch.argmax(similarity(delta, feats), dim=-1).unsqueeze(-1).expand_as(delta)
    # centroids = torch.gather(feats, dim=1, index=selected_ids)
    centroids = (delta + centroids) / 2
    return pi, centroids, gamma


def balanced_mean_shift(v_feats, g_feats, n_clus, tau, eps=1.0, iters=10, alpha=1):
    n_points = v_feats.shape[1]
    idx = torch.randperm(n_points)[:n_clus].tolist()
    v_cents = v_feats[:, idx]
    mu, nu = None, None
    gamma = None
    for _ in range(iters):
        pi, v_cents, gamma = one_step_mean_shift(v_feats, v_cents, mu, None, tau, eps)
        p2c_scores = similarity(v_feats, v_cents)
        b_pi = (1.0 - torch.pow(pi, alpha)) / (1.0 - pi).clamp(min=1e-3)
        ids = torch.argmax(p2c_scores, dim=-1)
        mu = b_pi.gather(dim=1, index=ids)
        mu = mu / torch.sum(mu, dim=-1, keepdim=True)
        nu = pi
    g_cents = gmm_params(gamma, g_feats)[1]

    return v_cents, g_cents, nu


def np_mean_shift(v_feats, g_feats, is_mean=False):
    # norm_feats = normalize_matrix(copy.deepcopy(v_feats), axis=1, norm='l2')
    db = DBSCAN(metric='cosine', eps=0.15, min_samples=20).fit(v_feats)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    v_centers = []
    g_centers = []

    for k in range(n_clusters_):
        # Get the points belonging to the current cluster
        if is_mean:
            v_center = np.mean(v_feats[labels == k], axis=0)
            g_center = np.mean(g_feats[labels == k], axis=0)
        else:
            v_center = v_feats[labels == k][0]
            g_center = g_feats[labels == k][0]
        v_centers.append(torch.from_numpy(v_center))
        g_centers.append(torch.from_numpy(g_center))

    v_centers = torch.stack(v_centers)
    g_centers = torch.stack(g_centers)

    return v_centers, g_centers, n_clusters_




