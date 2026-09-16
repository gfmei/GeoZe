"""Cut pursuit for superpoints -- a batched GPU rewrite, next to the CPU reference.

Cut pursuit (Landrieu & Obozinski, SIAM J. Imaging Sci. 2017) learns a piecewise-constant
approximation of a graph signal by minimising

    E(x) = sum_i ||x_i - y_i||^2  +  lambda * sum_{(i,j) in E} w_ij * [x_i != x_j]      (L0)

and it is the partition Superpoint Graph uses. The reference implementation this file follows
(github.com/truebelief/CutPursuit, used by CloudCompare's treeiso) is NumPy plus PyMaxflow: the
algorithm alternates a SPLIT step, which cuts every current component in two by an exact
max-flow, with a REDUCE step that recomputes component values.

The split is the part that does not port. Boykov-Kolmogorov max-flow is sequential and
per-component, so the reference is a Python loop over shapes and then over components.
Everything else in cut pursuit is a scatter-reduce and batches directly.

So the two solvers here share EVERYTHING except the binary step:

    split proposal   the principal direction of the within-component residual, by power
                     iteration, refined by 2-means -- identical in both
    binary solve     `maxflow`  exact min-cut per component, the reference behaviour
                     `icm`      parallel conditional modes over the whole batch at once
    accept           a split is kept only if it lowers E, evaluated exactly in both

That isolates the approximation: any quality difference between the two is the cost of replacing
an exact min-cut with a parallel local search, and the timing difference is what the rewrite buys.
Components at most double per round, so the outer loop is ~log2(n_sp) rounds either way.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath('../'))
from partseg.simple_geoze import (EPS, affinity, cue_features, gather_nb,  # noqa: E402
                                  knn, split_and_clean)


def gather_s(v, idx):
    B, N, k = idx.shape
    return v.gather(1, idx.reshape(B, N * k)).reshape(B, N, k)


# --------------------------------------------------------------------------- #
#  shared: component statistics and the split proposal                         #
# --------------------------------------------------------------------------- #
def comp_mean(y, lab, C):
    """[B,N,D] -> [B,C,D] mean of y within each component."""
    B, N, D = y.shape
    num = y.new_zeros(B, C, D).scatter_add_(1, lab.unsqueeze(-1).expand(-1, -1, D), y)
    cnt = y.new_zeros(B, C).scatter_add_(1, lab, torch.ones_like(lab, dtype=y.dtype))
    return num / cnt.clamp_min(1).unsqueeze(-1), cnt


def propose(y, lab, C, iters=8, kmeans_iters=5):
    """Binary proposal per component: split along the residual's principal direction, then
    two rounds of 2-means.  Returns b [B,N] in {0,1} and the two values v0, v1 [B,C,D]."""
    B, N, D = y.shape
    mu = comp_mean(y, lab, C)[0]
    r = y - mu.gather(1, lab.unsqueeze(-1).expand(-1, -1, D))
    d = F.normalize(torch.randn(B, C, D, generator=torch.Generator().manual_seed(0)).to(y), dim=-1)
    for _ in range(iters):                                   # power iteration on sum r r^T
        proj = (r * d.gather(1, lab.unsqueeze(-1).expand(-1, -1, D))).sum(-1, keepdim=True)
        d = torch.zeros_like(d).scatter_add_(
            1, lab.unsqueeze(-1).expand(-1, -1, D), r * proj)
        d = F.normalize(d, dim=-1)
    b = ((r * d.gather(1, lab.unsqueeze(-1).expand(-1, -1, D))).sum(-1) > 0).long()
    for _ in range(kmeans_iters):                            # 2-means inside each component
        key = lab * 2 + b
        v = comp_mean(y, key, 2 * C)[0]
        d0 = (y - v.gather(1, (lab * 2).unsqueeze(-1).expand(-1, -1, D))).pow(2).sum(-1)
        d1 = (y - v.gather(1, (lab * 2 + 1).unsqueeze(-1).expand(-1, -1, D))).pow(2).sum(-1)
        nb = (d1 < d0).long()
        if torch.equal(nb, b):
            break
        b = nb
    v = comp_mean(y, lab * 2 + b, 2 * C)[0]
    return b, v


def energy_terms(y, v, lab, b, w, idx, lam):
    """Per-component fidelity gain and cut cost of the proposed binary split."""
    B, N, D = y.shape
    C = v.shape[1] // 2
    key = lab * 2 + b
    fid = (y - v.gather(1, key.unsqueeze(-1).expand(-1, -1, D))).pow(2).sum(-1)
    mu = comp_mean(y, lab, C)[0]
    fid0 = (y - mu.gather(1, lab.unsqueeze(-1).expand(-1, -1, D))).pow(2).sum(-1)
    same = lab.unsqueeze(-1) == gather_s(lab, idx)
    cut = 0.5 * lam * (w * same * (b.unsqueeze(-1) != gather_s(b, idx))).sum(-1)
    d_fid = torch.zeros(B, C, device=y.device, dtype=y.dtype).scatter_add_(1, lab, fid - fid0)
    d_cut = torch.zeros(B, C, device=y.device, dtype=y.dtype).scatter_add_(1, lab, cut)
    return d_fid + d_cut                                     # < 0 means the split pays


# --------------------------------------------------------------------------- #
#  the two binary solvers                                                      #
# --------------------------------------------------------------------------- #
def refine_icm(y, v, lab, b, w, idx, lam, iters=10):
    """Parallel conditional modes: every node picks the cheaper label at once, batched.

    A Jacobi sweep can oscillate, so the labelling is only kept when the exact energy drops.
    """
    B, N, D = y.shape
    best, best_e = b, energy_terms(y, v, lab, b, w, idx, lam).sum()
    same = lab.unsqueeze(-1) == gather_s(lab, idx)
    u0 = (y - v.gather(1, (lab * 2).unsqueeze(-1).expand(-1, -1, D))).pow(2).sum(-1)
    u1 = (y - v.gather(1, (lab * 2 + 1).unsqueeze(-1).expand(-1, -1, D))).pow(2).sum(-1)
    for _ in range(iters):
        nb = gather_s(b, idx)
        p0 = lam * (w * same * (nb == 1)).sum(-1)
        p1 = lam * (w * same * (nb == 0)).sum(-1)
        new = ((u1 + p1) < (u0 + p0)).long()
        if torch.equal(new, b):
            break
        b = new
        e = energy_terms(y, v, lab, b, w, idx, lam).sum()
        if e < best_e:
            best, best_e = b, e
    return best


def refine_maxflow(y, v, lab, b, w, idx, lam):
    """Exact min-cut per component, the reference behaviour.  CPU, and a loop over components."""
    import maxflow
    B, N, D = y.shape
    dev = y.device
    yb, vb, lb = y.cpu().numpy(), v.cpu().numpy(), lab.cpu().numpy()
    wb, ib, bb = w.cpu().numpy(), idx.cpu().numpy(), b.cpu().numpy()
    out = bb.copy()
    for s in range(B):
        for c in np.unique(lb[s]):
            m = np.where(lb[s] == c)[0]
            if m.size < 2:
                continue
            pos = -np.ones(N, np.int64)
            pos[m] = np.arange(m.size)
            u0 = ((yb[s][m] - vb[s][2 * c]) ** 2).sum(-1)
            u1 = ((yb[s][m] - vb[s][2 * c + 1]) ** 2).sum(-1)
            g = maxflow.Graph[float](m.size, m.size * ib.shape[-1])
            nodes = g.add_nodes(m.size)
            g.add_grid_tedges(nodes, u1, u0)                 # source=label0, sink=label1
            src = np.repeat(m, ib.shape[-1])
            dst = ib[s][m].reshape(-1)
            cap = lam * wb[s][m].reshape(-1)
            keep = (lb[s][dst] == c) & (pos[dst] > pos[src])
            ia, ja, ca = pos[src[keep]], pos[dst[keep]], cap[keep]
            if ia.size:
                try:
                    g.add_edges(ia, ja, ca, ca)              # vectorised where available
                except AttributeError:
                    for a_, z_, c_ in zip(ia, ja, ca):
                        g.add_edge(int(a_), int(z_), float(c_), float(c_))
            g.maxflow()
            out[s][m] = np.array([g.get_segment(int(p)) for p in range(m.size)])
    return torch.from_numpy(out).to(dev)


# --------------------------------------------------------------------------- #
#  cut pursuit                                                                 #
# --------------------------------------------------------------------------- #
def compact(lab):
    """[B,N] arbitrary ids -> contiguous 0-based ids PER SHAPE, with no loop over shapes.

    `unique` sorts, and the key is shape-major, so every id of shape b is contiguous and below
    every id of shape b+1; subtracting each row's minimum makes them 0-based independently.
    """
    B, N = lab.shape
    key = torch.arange(B, device=lab.device).unsqueeze(1) * (int(lab.max()) + 1) + lab
    g = torch.unique(key, return_inverse=True)[1].reshape(B, N)
    return g - g.amin(1, keepdim=True)


def cutpursuit(y, w, idx, lam, rounds=7, solver='icm'):
    """L0 cut pursuit on a batch of kNN graphs.  Returns component labels [B,N].

    The outer loop is over ROUNDS, not over shapes or components: components at most double each
    round, so ~log2(target) rounds cover any partition size, and every shape and every component
    is split in the same pass.
    """
    B, N, _ = y.shape
    lab = torch.zeros(B, N, dtype=torch.long, device=y.device)
    for _ in range(rounds):
        C = int(lab.max()) + 1
        b, v = propose(y, lab, C)
        b = refine_icm(y, v, lab, b, w, idx, lam) if solver == 'icm' \
            else refine_maxflow(y, v, lab, b, w, idx, lam)
        keep = (energy_terms(y, v, lab, b, w, idx, lam) < 0).gather(1, lab)
        if not keep.any():                                   # no split lowers E anywhere
            break
        lab = compact(torch.where(keep, lab * 2 + b, lab * 2))
    return lab


def cutpursuit_superpoints(xyz, nrm, fpfh, lam=1.0, k=10, rounds=7, solver='icm'):
    """Coordinates + normals + FPFH -> flat [B*N] region ids, grouped by shape."""
    idx = knn(xyz, k)
    y, g, n = cue_features(xyz, nrm, fpfh, idx)
    w = affinity(xyz, n, g, idx)
    lab = cutpursuit(y, w, idx, lam, rounds, solver)
    return split_and_clean(lab, idx)


# --------------------------------------------------------------------------- #
#  comparison                                                                  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def main():
    from libs.lib_metric import seg_num
    from partseg.partmodel.eval_v2 import load_class, oracle_pred
    from partseg.partmodel.post_search import cat2id
    from partseg.probe_superpoints import boundary, quality

    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', nargs='+', default=['airplane', 'chair', 'table'])
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--lam', type=float, nargs='+', default=[1.0])
    ap.add_argument('--rounds', type=int, nargs='+', default=[5, 6, 7])
    ap.add_argument('--solvers', nargs='+', default=['icm', 'maxflow'])
    ap.add_argument('--bs', type=int, default=10)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else 'cpu'

    acc = {}
    for c in a.classes:
        d = load_class(c, 'ViT-B/16')
        n = min(a.limit, d['pc'].shape[0])
        P = seg_num[cat2id[c]]
        print(f'\n===== {c} ({n} shapes)')
        print(f'{"solver":>8s}{"lam":>6s}{"rnds":>6s}{"regions":>9s}{"oracleIoU":>11s}'
              f'{"BR":>8s}{"ms/shape":>10s}')
        for solver in a.solvers:
            for lam in a.lam:
              for rnd in a.rounds:
                rows, ms = [], []
                for s in range(0, n, a.bs):
                    xyz, nrm, fp = (d[q][s:s + a.bs].to(dev) for q in ('pc', 'normal', 'fpfh'))
                    lab = d['label'][s:s + a.bs].to(dev).reshape(-1)
                    B = xyz.shape[0]
                    if dev.startswith('cuda'):
                        torch.cuda.synchronize()
                    t0 = time.time()
                    seg = cutpursuit_superpoints(xyz, F.normalize(nrm, dim=-1), fp, lam,
                                                 rounds=rnd, solver=solver)
                    if dev.startswith('cuda'):
                        torch.cuda.synchronize()
                    ms.append((time.time() - t0) * 1e3 / B)
                    rows.append(((int(seg.max()) + 1) / B,)
                                + quality(seg, lab, knn(xyz, 10), P))
                r = np.mean(np.array(rows), axis=0)
                t = float(np.mean(ms[1:] if len(ms) > 1 else ms))
                acc.setdefault((solver, lam, rnd), []).append((r[0], r[1], r[3], t))
                print(f'{solver:>8s}{lam:>6.1f}{rnd:>6d}{r[0]:>9.1f}{r[1]:>11.2f}'
                      f'{r[3]:>8.2f}{t:>10.1f}', flush=True)

    print(f'\n===== mean over classes\n{"solver":>8s}{"lam":>6s}{"rnds":>6s}{"regions":>9s}'
          f'{"oracleIoU":>11s}{"BR":>8s}{"ms/shape":>10s}')
    for (solver, lam, rnd), v in acc.items():
        r = np.mean(np.array(v), axis=0)
        print(f'{solver:>8s}{lam:>6.1f}{rnd:>6d}{r[0]:>9.1f}{r[1]:>11.2f}{r[2]:>8.2f}{r[3]:>10.1f}')


if __name__ == '__main__':
    main()
