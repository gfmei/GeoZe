"""Point-graph primitives shared by the object and scene pipelines.

These are the operations both `partseg/` and `semseg/` need and neither owns: build a neighbour
graph, gather over it, reduce over a segmentation, and clean a partition up. Keeping one
definition here means the two pipelines cannot drift apart on, say, what a segment mean is.
"""
import torch

EPS = 1e-8


# --------------------------------------------------------------------------- #
#  neighbour graph                                                             #
# --------------------------------------------------------------------------- #
def knn(xyz, k):
    """[B,N,k] indices of the k nearest neighbours, self excluded.

    A dense distance matrix, which is the right choice at object scale: at N=2048 it costs
    0.19 ms/shape and an approximate graph measurably loses accuracy for no time saved. Scenes
    are two orders of magnitude larger and use `semseg/semmodel/curve.py` instead.
    """
    return torch.cdist(xyz, xyz).topk(k + 1, dim=-1, largest=False)[1][:, :, 1:]


def gather_nb(x, idx):
    """x [B,N,D], idx [B,N,k] -> [B,N,k,D]: the neighbours' vectors."""
    B, N, k = idx.shape
    return x.gather(1, idx.reshape(B, N * k, 1).expand(-1, -1, x.shape[-1])).reshape(B, N, k, -1)


def gather_s(v, idx):
    """v [B,N], idx [B,N,k] -> [B,N,k]: the neighbours' scalars."""
    B, N, k = idx.shape
    return v.gather(1, idx.reshape(B, N * k)).reshape(B, N, k)


# --------------------------------------------------------------------------- #
#  segment reduction                                                           #
# --------------------------------------------------------------------------- #
def seg_mean(vals, seg, S, w=None):
    """Weighted per-segment mean of [N,D] -> [S,D].  w: [N] or None."""
    D = vals.shape[1]
    num = vals.new_zeros(S, D)
    den = vals.new_zeros(S, 1)
    ww = torch.ones_like(vals[:, :1]) if w is None else w.unsqueeze(1).to(vals.dtype)
    num.index_add_(0, seg, vals * ww)
    den.index_add_(0, seg, ww)
    return num / den.clamp_min(EPS)


# --------------------------------------------------------------------------- #
#  partition cleanup                                                           #
# --------------------------------------------------------------------------- #
def split_and_clean(lab, idx, min_size=4):
    """Cut every cluster into its connected components, absorb fragments, return flat ids.

    A cluster is a set of points that agreed in some feature space, and nothing so far forces it
    to be one connected piece; a region made of two distant blobs pools two different parts
    together. Connected components are found by min-label propagation with pointer jumping, which
    converges in O(log N) rounds, and the ids come back flat and grouped by shape so a segment
    reduction can run over the whole batch at once.
    """
    B, N, k = idx.shape
    flat = idx.reshape(B, -1)
    same = lab.unsqueeze(2) == lab.gather(1, flat).reshape(B, N, k)
    comp = torch.arange(N, device=lab.device).expand(B, N).clone()
    for _ in range(int(N).bit_length()):
        pull = comp.gather(1, flat).reshape(B, N, k).masked_fill(~same, N).min(-1)[0]
        push = comp.unsqueeze(2).expand(B, N, k).masked_fill(~same, N).reshape(B, -1)
        new = torch.minimum(comp, pull).scatter_reduce(1, flat, push, 'amin')
        new = new.gather(1, new)
        if torch.equal(new, comp):
            break
        comp = new

    key = lab * N + comp
    key = torch.arange(B, device=lab.device).unsqueeze(1) * (int(key.max()) + 1) + key
    seg = torch.unique(key, return_inverse=True)[1].reshape(-1)

    nb = (idx + (torch.arange(B, device=idx.device) * N).view(B, 1, 1)).reshape(B * N, k)
    for _ in range(3):
        cnt = torch.bincount(seg, minlength=int(seg.max()) + 1)
        small = cnt[seg] < min_size
        if not small.any():
            break
        big = ~small[nb]
        first = big.to(torch.uint8).argmax(1)
        cand = seg[nb.gather(1, first.unsqueeze(1)).squeeze(1)]
        seg = torch.where(small & big.any(1), cand, seg)
    return torch.unique(seg, return_inverse=True)[1]
