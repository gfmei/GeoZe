"""Shared segment-reduction helpers for the scene-level GeoZe modules."""
import torch
import torch.nn.functional as F

EPS = 1e-8


def seg_mean(vals, seg, S, w=None):
    """Weighted per-segment mean of [N,D] -> [S,D].  w: [N] or None."""
    D = vals.shape[1]
    num = vals.new_zeros(S, D)
    den = vals.new_zeros(S, 1)
    ww = torch.ones_like(vals[:, :1]) if w is None else w.unsqueeze(1).to(vals.dtype)
    num.index_add_(0, seg, vals * ww)
    den.index_add_(0, seg, ww)
    return num / den.clamp_min(EPS)


def batched_eigh(C, chunk=32768):
    """Eigen-decomposition of many small symmetric matrices, [B,3,3] -> eigenvectors [B,3,3].

    cuSOLVER's batched syev rejects very large batches (a 278k-point LiDAR scan trips it), so
    the batch is chunked, with a CPU fallback if the GPU path still refuses.  Ascending order,
    so [:, :, 0] is the smallest eigenvector and [:, :, -1] the largest.
    """
    C = torch.nan_to_num(C.double())
    out = torch.empty_like(C)
    for s in range(0, C.shape[0], chunk):
        blk = C[s:s + chunk]
        try:
            out[s:s + chunk] = torch.linalg.eigh(blk)[1]
        except Exception:                                    # cuSOLVER refusal -> CPU
            out[s:s + chunk] = torch.linalg.eigh(blk.cpu())[1].to(C.device)
    return out


def seg_normal(nrm, seg, S):
    """Orientation-free mean normal per segment: top eigenvector of sum(n n^T).

    PCA/estimated normals are UNSIGNED, so a plain mean cancels on a curved patch and gives a
    meaningless direction; the outer-product eigenvector is invariant to per-point sign flips.
    """
    nn = F.normalize(nrm, dim=-1)
    M = seg_mean((nn.unsqueeze(2) * nn.unsqueeze(1)).reshape(-1, 9), seg, S).reshape(S, 3, 3)
    return batched_eigh(M)[:, :, -1].to(nrm.dtype)


def seg_softmax(e, src, S):
    """Softmax of edge scores within each source segment's neighbourhood."""
    mx = torch.full((S,), -1e9, device=e.device, dtype=e.dtype).index_reduce_(0, src, e, 'amax')
    ex = torch.exp(e - mx[src])
    den = torch.zeros(S, device=e.device, dtype=e.dtype).index_add_(0, src, ex)
    return ex / den[src].clamp_min(EPS)


def estimate_normals(xyz, i, j):
    """Per-point normals by PCA over a neighbour graph, entirely on GPU.

    (i, j) is the same point-pair graph the region merging already builds, so LiDAR datasets
    that ship no normals (nuScenes) need no open3d pre-pass at all: the smallest eigenvector
    of each point's neighbour covariance is its normal.  Sign is arbitrary, which is fine —
    every consumer here uses |n . n| or the outer-product mean (see seg_normal).
    """
    N = xyz.shape[0]
    mu = torch.zeros(N, 3, device=xyz.device, dtype=xyz.dtype)
    cnt = torch.zeros(N, 1, device=xyz.device, dtype=xyz.dtype)
    mu.index_add_(0, i, xyz[j])
    cnt.index_add_(0, i, torch.ones_like(xyz[j][:, :1]))
    mu = mu / cnt.clamp_min(1)
    dv = xyz[j] - mu[i]
    C = torch.zeros(N, 9, device=xyz.device, dtype=xyz.dtype)
    C.index_add_(0, i, (dv.unsqueeze(2) * dv.unsqueeze(1)).reshape(-1, 9))
    C = (C / cnt.clamp_min(1)).reshape(N, 3, 3)
    C = C + torch.eye(3, device=xyz.device, dtype=xyz.dtype) * 1e-9      # keep eigh well-posed
    n = batched_eigh(C)[:, :, 0].to(xyz.dtype)                           # smallest eigenvector
    lone = cnt.squeeze(1) < 3                                            # too few neighbours
    if lone.any():
        n[lone] = torch.tensor([0.0, 0.0, 1.0], device=xyz.device, dtype=xyz.dtype)
    return F.normalize(n, dim=-1)


def robust_scale(d, m, mode='mean'):
    """Per-region adaptive temperature of a distance block [B,A,C] -> [B,1,1].

    Floored at 1% of the global scale: a region uniform in one cue has ~0 spread there, and
    dividing by it would let float noise dominate the affinity.
    """
    if mode == 'median':
        big = d.masked_fill(~m, float('inf')).flatten(1)
        cnt = m.flatten(1).sum(1).clamp_min(1)
        srt, _ = big.sort(dim=1)
        s = srt.gather(1, (cnt // 2).unsqueeze(1).clamp(max=srt.shape[1] - 1)).squeeze(1)
        s = torch.where(torch.isfinite(s), s, torch.zeros_like(s))
    else:
        s = (d * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp_min(1)
    floor = 1e-2 * (d * m).sum() / m.sum().clamp_min(1)
    return s.clamp_min(floor.clamp_min(EPS)).view(-1, 1, 1)
