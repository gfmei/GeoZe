"""Multi-curve VOTING kNN graph — the O(N log N) stand-in for a KD-tree.

A space-filling curve maps 3D to 1D while preserving most locality, so a window in the sorted
order is a cheap approximate neighbourhood.  Any single curve also produces spurious links
wherever its window straddles a curve discontinuity, and taking the UNION over curves keeps
every one of those.  Voting fixes the precision side: a pair is a neighbour only if several
independent curves place the two points close together, and a discontinuity in one curve
essentially never lines up with a discontinuity in another.

    votes(i,j) = #curves whose window contains the pair
    keep       = votes >= min_votes, then per point the top-k by (votes, -distance)

Eight curves: {z, z-trans, hilbert, hilbert-trans} x {origin, half-cell shift}.  The shift
decorrelates the quantisation boundaries, which is where most artefacts come from.

Measured on ScanNet val: ~35% faster than scipy's cKDTree (71 vs 109 ms/scene) at equal
accuracy on full val.  The KD-tree remains the default in best_param because it was very
slightly ahead there (+0.43 vs +0.21 mIoU); use `--curve` to switch.
"""
import torch

from libs.serialization import encode

ORDERS = ('z', 'z-trans', 'hilbert', 'hilbert-trans')
CURVES = tuple((o, s) for s in (0.0, 0.5) for o in ORDERS)


def serial_order(xyz, order, shift, depth, voxel):
    g = ((xyz - xyz.min(0).values) / voxel + shift).floor().long().clamp_(0, 2 ** depth - 1)
    code = encode(g, torch.zeros(g.shape[0], dtype=torch.long, device=g.device),
                  depth=depth, order=order)
    return torch.argsort(code)


def knn_graph(xyz, k=8, curves=CURVES, min_votes=4, window=0, depth=14, voxel=0.02, max_dist=0.3):
    """Approximate symmetric kNN graph by multi-curve voting.  Returns (i, j) point pairs."""
    N = xyz.shape[0]
    w = window or max(1, k // 2)
    src, dst = [], []
    for order, sh in curves:
        o = serial_order(xyz, order, sh, depth, voxel)
        for s in range(1, w + 1):
            src.append(o[: N - s]); dst.append(o[s:])
    i = torch.cat(src); j = torch.cat(dst)
    key = torch.minimum(i, j) * N + torch.maximum(i, j)
    key, _ = torch.sort(key)
    first = torch.ones_like(key, dtype=torch.bool)
    first[1:] = key[1:] != key[:-1]
    ids = torch.nonzero(first, as_tuple=True)[0]
    votes = torch.diff(torch.cat([ids, ids.new_tensor([key.numel()])]))     # run length = #curves
    key = key[first]
    i, j = key // N, key % N

    d2 = (xyz[i] - xyz[j]).pow(2).sum(-1)
    keep = (votes >= min_votes) & (d2 <= max_dist ** 2)
    i, j, d2, votes = i[keep], j[keep], d2[keep], votes[keep]

    if k > 0:                                    # per-point top-k by (votes desc, distance asc)
        i2 = torch.cat([i, j]); j2 = torch.cat([j, i])
        sc = torch.cat([votes, votes]).float() - torch.cat([d2, d2]) / max(max_dist ** 2, 1e-9)
        o = torch.argsort(sc, descending=True)
        o = o[torch.argsort(i2[o], stable=True)]
        rank = torch.empty_like(o)
        rank[o] = torch.arange(o.numel(), device=o.device)
        firstpos = torch.full((N,), o.numel(), dtype=torch.long, device=o.device)
        firstpos = firstpos.index_reduce_(0, i2, rank, 'amin')
        sel = (rank - firstpos[i2]) < k
        return i2[sel], j2[sel]
    return torch.cat([i, j]), torch.cat([j, i])


def kdtree_graph(xyz_np, k, device):
    """Exact kNN pairs via scipy cKDTree, as (i, j) on `device`."""
    import numpy as np
    from scipy.spatial import cKDTree
    kn = cKDTree(xyz_np).query(xyz_np, k=k + 1, workers=-1)[1][:, 1:]
    N = xyz_np.shape[0]
    i = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
    j = torch.from_numpy(np.ascontiguousarray(kn)).to(device).reshape(-1)
    return i, j
