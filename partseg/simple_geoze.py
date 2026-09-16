"""SimpleGeoZe -- superpoint pooling and label propagation, in one file.

This is the whole method that survived measurement, written linearly and with no options.
`partmodel/` carries four partitions, four eigensolvers, a merging stage and an attention stage
because each had to be measured; almost none of it paid, and what is left is short enough to read
in one sitting:

    1. a geometry-only partition          k-means over position + normals + FPFH, then a few
                                          rounds of affinity-weighted boundary refinement
    2. pool the VLM feature per region    a masked mean of the unit features
    3. classify the REGIONS               S x D against the text table, not N x D
    4. propagate the label to points      exact, since argmax of a broadcast vector is the
                                          broadcast of its argmax

Nothing here touches the VLM feature until step 2, so a noisy feature can never corrupt the
support it is pooled over.

What was measured and deliberately left out (numbers in partseg/README.md): hierarchical merging
(-1.81 class-mIoU on parts), intra- and inter-region attention, the spectral solve (+0.3 for 3x
the time), Nystrom, LOBPCG, and multi-curve kNN.

    python simple_geoze.py --classchoice all
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath('../'))

EPS = 1e-8


# --------------------------------------------------------------------------- #
#  partition                                                                   #
# --------------------------------------------------------------------------- #
def knn(xyz, k):
    """[B,N,k] indices of the k nearest neighbours, self excluded.  Dense at 2k points."""
    return torch.cdist(xyz, xyz).topk(k + 1, dim=-1, largest=False)[1][:, :, 1:]


def gather_nb(x, idx):
    """x [B,N,D], idx [B,N,k] -> [B,N,k,D]."""
    B, N, k = idx.shape
    return x.gather(1, idx.reshape(B, N * k, 1).expand(-1, -1, x.shape[-1])).reshape(B, N, k, -1)


def cue_features(xyz, nrm, fpfh, idx):
    """The three per-point cues, each scaled so one mean kNN step is one unit.

    Normals enter through the 6 unique entries of n n^T, not as n: an estimated normal's SIGN is
    arbitrary, and the outer product is invariant to flipping it, so a sign flip cannot split a
    region.  FPFH is L1-normalised and square-rooted, which makes the inner product between two
    of them the Bhattacharyya coefficient -- i.e. it behaves like a cosine.
    """
    h = fpfh.clamp_min(0)
    g = (h / h.sum(-1, keepdim=True).clamp_min(EPS)).sqrt()
    n = F.normalize(nrm, dim=-1)
    r2 = 2.0 ** 0.5
    nn = torch.stack([n[..., 0] ** 2, n[..., 1] ** 2, n[..., 2] ** 2,
                      r2 * n[..., 0] * n[..., 1], r2 * n[..., 0] * n[..., 2],
                      r2 * n[..., 1] * n[..., 2]], dim=-1)
    out = []
    for c in (xyz, nn, g):
        step = (c.unsqueeze(2) - gather_nb(c, idx)).norm(dim=-1).mean(dim=(1, 2)).clamp_min(EPS)
        out.append(c / step.view(-1, 1, 1))
    return torch.cat(out, dim=-1), g, n


def morton_seed(xyz, K, depth=10):
    """K well-spread seeds from ONE sort, with no loop over seeds.

    Farthest-point sampling is the usual choice but it is sequential in K: at 128 seeds that is
    128 dependent GPU launches. Points adjacent in Z-order are spatially close, so a stride
    through that order covers the shape evenly for one `argsort`. The only loop here runs over
    the 10 BITS of the grid, independent of K and of the number of points.
    """
    B, N, _ = xyz.shape
    g = xyz - xyz.amin(1, keepdim=True)
    g = (g / g.amax(dim=(1, 2)).clamp_min(EPS).view(B, 1, 1) * (2 ** depth - 1))
    g = g.long().clamp_(0, 2 ** depth - 1)
    code = torch.zeros(B, N, dtype=torch.long, device=xyz.device)
    for b in range(depth):
        m = 1 << b
        code |= (((g[..., 0] & m) << (2 * b)) | ((g[..., 1] & m) << (2 * b + 1))
                 | ((g[..., 2] & m) << (2 * b + 2)))
    order = code.argsort(1)
    return order.gather(1, torch.linspace(0, N - 1, K, device=xyz.device).long()
                        .unsqueeze(0).expand(B, K))


def fps_seed(xyz, K):
    """Farthest-point seeds. Sequential in K, kept because it is the quality reference."""
    B, N, _ = xyz.shape
    seeds = torch.zeros(B, K, dtype=torch.long, device=xyz.device)
    far = (xyz - xyz.mean(1, keepdim=True)).norm(dim=-1).argmax(1)
    dist = torch.full((B, N), 1e10, device=xyz.device, dtype=xyz.dtype)
    for _ in range(K):
        seeds[:, _] = far
        d = (xyz - xyz.gather(1, far.view(B, 1, 1).expand(-1, -1, 3))).pow(2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(1)
    return seeds


def kmeans(X, xyz, K, iters=20, seed='fps'):
    """Batched k-means. Every step is one batched op; the loop is over Lloyd rounds only."""
    seeds = morton_seed(xyz, K) if seed == 'curve' else fps_seed(xyz, K)
    C = X.gather(1, seeds.unsqueeze(-1).expand(-1, -1, X.shape[-1]))
    lab = None
    for it in range(iters):
        new = torch.cdist(X, C).argmin(-1)
        if lab is not None and torch.equal(new, lab):
            break
        lab = new
        oh = F.one_hot(lab, K).to(X.dtype)
        cnt = oh.sum(1)
        C = torch.where((cnt > 0).unsqueeze(-1),
                        torch.einsum('bnk,bnd->bkd', oh, X) / cnt.clamp_min(1).unsqueeze(-1), C)
    return lab


def orient(xyz, n, idx):
    """Resolve the arbitrary sign of estimated normals so neighbours agree.

    An estimator returns normals up to sign, and the concavity test below is meaningless without
    a consistent orientation.  Maximising sum_ij s_i s_j (n_i.n_j) over the graph is an Ising
    problem; sign iteration seeded outward from the centroid settles it in about five rounds.
    """
    B, N, k = idx.shape
    s = torch.sign((n * (xyz - xyz.mean(1, keepdim=True))).sum(-1))
    s = torch.where(s == 0, torch.ones_like(s), s)
    w = (n.unsqueeze(2) * gather_nb(n, idx)).sum(-1)
    for _ in range(8):                              # settles in about five
        new = torch.sign((w * s.gather(1, idx.reshape(B, -1)).reshape(B, N, k)).sum(-1))
        new = torch.where(new == 0, s, new)
        if torch.equal(new, s):
            break
        s = new
    return n * s.unsqueeze(-1)


def affinity(xyz, n, g, idx):
    """Per-edge affinity over four cues, each divided by its own mean so nothing needs tuning.

    The fourth is CONCAVITY: object parts meet at concave seams (seat/leg, wing/body), so a
    concave edge is penalised and a convex one is free.
    """
    m = orient(xyz, n, idx)
    dx = xyz.unsqueeze(2) - gather_nb(xyz, idx)
    d = [dx.pow(2).sum(-1),
         (1 - (n.unsqueeze(2) * gather_nb(n, idx)).sum(-1).abs()).clamp_min(0),
         (1 - (g.unsqueeze(2) * gather_nb(g, idx)).sum(-1)).clamp_min(0),
         F.relu(-((m.unsqueeze(2) - gather_nb(m, idx)) * F.normalize(dx, dim=-1)).sum(-1))]
    e = torch.zeros_like(d[0])
    for v in d:
        e = e - v / v.mean(dim=(1, 2), keepdim=True).clamp_min(EPS)
    return torch.exp(e)


def spectral(w, idx, n_sp, iters=50, over=16):
    """Normalised-cut embedding without ever forming the N x N matrix.

    The textbook route builds D^-1/2 W D^-1/2 and decomposes it, which at N=2048 costs 35 of the
    pipeline's 38 ms to keep 3% of its output.  Here the operator is only ever APPLIED: W x is a
    gather over the kNN edges and W^T x the matching scatter, so nothing N x N is allocated.
    Subspace iteration with Cholesky-QR extracts the leading block, and a closing Rayleigh-Ritz
    rotates it onto the eigenvectors -- which matters, because k-means is not invariant to a
    rotation of the embedding.
    """
    B, N, k = idx.shape
    dev = w.device

    # One sparse operator over the FLATTENED batch. The batch graph is block diagonal -- no edge
    # crosses shapes -- so a single [B*N, B*N] matrix holds all of them and one `sparse.mm`
    # replaces a [B,N,k,m] gather that would be materialised on every iteration (at m=144 that
    # array is 44M elements, and the iteration is memory-bound on it).
    off = (torch.arange(B, device=dev) * N).view(B, 1, 1)
    src = (torch.arange(N, device=dev).view(1, N, 1) + off).expand(B, N, k).reshape(-1)
    dst = (idx + off).reshape(-1)
    val = 0.5 * w.reshape(-1)
    ii = torch.cat([src, dst])                      # both directions; coalesce sums duplicates,
    jj = torch.cat([dst, src])                      # which is exactly (W + W^T)/2
    vv = torch.cat([val, val])

    def build(v):
        return torch.sparse_coo_tensor(torch.stack([ii, jj]), v, (B * N, B * N)).coalesce()

    deg = build(vv) @ torch.ones(B * N, 1, device=dev, dtype=w.dtype)
    r = deg.squeeze(-1).clamp_min(EPS).rsqrt()
    A = build(vv * r[ii] * r[jj])                   # D^-1/2 W D^-1/2, symmetric by construction

    def apply(V):                                   # [B,N,m] -> [B,N,m]
        return (A @ V.reshape(B * N, -1)).view(B, N, -1)

    def orth(M):                                    # Cholesky-QR: two thin matmuls, tiny chol
        # in the working dtype, with a jitter scaled to the Gram diagonal; a Cholesky failure
        # (near-rank-deficient block) falls back to float64 rather than to a slower default,
        # because doing every iteration in float64 costs more than the whole rest of the method
        G = M.transpose(1, 2) @ M
        eye = torch.eye(G.shape[-1], device=M.device, dtype=G.dtype).unsqueeze(0)
        jit = 1e-6 * torch.diagonal(G, dim1=1, dim2=2).mean(-1).clamp_min(EPS).view(-1, 1, 1)
        try:
            L = torch.linalg.cholesky(G + jit * eye)
        except Exception:                           # noqa: BLE001
            L = torch.linalg.cholesky((G + jit * eye).double()).to(G.dtype)
        # Measured: an explicit inv(L) plus a matmul, which looks cheaper on paper, is SLOWER
        # here (44.8 vs 38.9 ms/shape at m=144) -- the batched triangular solve wins.
        return torch.linalg.solve_triangular(L, M.transpose(1, 2), upper=False).transpose(1, 2)

    m = min(N, n_sp + over)
    gen = torch.Generator(device='cpu').manual_seed(0)
    X = orth(torch.randn(B, N, m, generator=gen).to(w.device, w.dtype))
    prev = None
    for it in range(iters):
        X = orth(apply(X) + X)                      # (S + I) keeps the spectrum positive
        if it % 5 == 4:                             # stop once the Ritz values settle
            T = X.transpose(1, 2) @ apply(X)
            ev = torch.linalg.eigvalsh(0.5 * (T + T.transpose(1, 2)))
            if prev is not None and (ev - prev).abs().max() < 1e-4:
                break
            prev = ev
    T = X.transpose(1, 2) @ apply(X)
    ev, U = torch.linalg.eigh(0.5 * (T + T.transpose(1, 2)))
    return F.normalize((X @ U)[:, :, -n_sp:], dim=-1).contiguous()


def relabel(lab, w, idx, K, rounds=3):
    """Snap region borders onto the weak edges of the graph.

    k-means places each point independently, so borders come out ragged and cut across strong
    edges.  Each round every point takes the label its strongest neighbours carry.
    """
    B, N, k = idx.shape
    for _ in range(rounds):
        sc = torch.zeros(B, N, K, device=lab.device, dtype=w.dtype)
        sc.scatter_add_(2, lab.gather(1, idx.reshape(B, -1)).reshape(B, N, k), w)
        new = sc.argmax(-1)
        if torch.equal(new, lab):
            break
        lab = new
    return lab


def split_and_clean(lab, idx, min_size=4):
    """Cut every cluster into its connected components, absorb fragments, return flat ids.

    A cluster is a set of points that agreed in cue space; nothing so far forces it to be one
    connected piece, and a region made of two distant blobs pools two different parts together.
    """
    B, N, k = idx.shape
    flat = idx.reshape(B, -1)
    same = lab.unsqueeze(2) == lab.gather(1, flat).reshape(B, N, k)
    comp = torch.arange(N, device=lab.device).expand(B, N).clone()
    for _ in range(int(N).bit_length()):            # pointer jumping: O(log N), not O(N)
        pull = comp.gather(1, flat).reshape(B, N, k).masked_fill(~same, N).min(-1)[0]
        push = comp.unsqueeze(2).expand(B, N, k).masked_fill(~same, N).reshape(B, -1)
        new = torch.minimum(comp, pull).scatter_reduce(1, flat, push, 'amin')
        new = new.gather(1, new)                                     # pointer jumping
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


def superpoints(xyz, nrm, fpfh, n_sp=32, k=10, rounds=3, part='kmeans', iters=50,
                seed='fps'):
    """[B,N] coordinates + normals + FPFH -> flat [B*N] region ids, grouped by shape.

    `part='kmeans'` clusters the cues directly; `part='spectral'` clusters a normalised-cut
    embedding of the same cue affinity, which is a better partition at a matched region count
    and about three times the cost.
    """
    idx = knn(xyz, k)
    X, g, n = cue_features(xyz, nrm, fpfh, idx)
    w = affinity(xyz, n, g, idx) if (rounds or part == 'spectral') else None
    lab = kmeans(spectral(w, idx, n_sp, iters), xyz, n_sp, seed=seed) \
        if part == 'spectral' else kmeans(X, xyz, n_sp, seed=seed)
    if rounds:
        lab = relabel(lab, w, idx, n_sp, rounds)
    return split_and_clean(lab, idx)


# --------------------------------------------------------------------------- #
#  pool, classify, propagate                                                   #
# --------------------------------------------------------------------------- #
def simple_geoze(xyz, nrm, fpfh, feat, text, n_sp=32, k=10, rounds=3, part='kmeans',
                 seed='fps'):
    """Per-point part labels [B,N].  `feat` [B,N,D] VLM features, `text` [C,D] L2-normalised."""
    B, N, D = feat.shape
    seg = superpoints(xyz, nrm, fpfh, n_sp, k, rounds, part, seed=seed)
    S = int(seg.max()) + 1

    f = F.normalize(feat.reshape(B * N, D), dim=-1)
    valid = (feat.reshape(B * N, D).norm(dim=-1) > 0).to(f.dtype)      # unseen points: no feature
    num = f.new_zeros(S, D).index_add_(0, seg, f * valid.unsqueeze(1))
    den = f.new_zeros(S, 1).index_add_(0, seg, valid.unsqueeze(1))
    z = F.normalize(num / den.clamp_min(EPS), dim=-1)                  # [S,D] region features

    return (z @ text.t()).argmax(-1)[seg].view(B, N)                   # classify, then propagate


# --------------------------------------------------------------------------- #
#  evaluation                                                                  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def main():
    from libs.lib_metric import calculate_shape_IoU, seg_num
    from partseg.partclip import clip
    from partseg.partmodel.best_param import best_vweight
    from partseg.partmodel.eval_v2 import load_class
    from partseg.partmodel.post_search import cat2id, textual_encoder
    from partseg.rendering.unprojection import vanilla_upprojection

    ap = argparse.ArgumentParser()
    ap.add_argument('--classchoice', default='all')
    ap.add_argument('--n_sp', type=int, default=32)
    ap.add_argument('--knn', type=int, default=10)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--part', default='kmeans', choices=['kmeans', 'spectral'])
    ap.add_argument('--seed', default='fps', choices=['fps', 'curve'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--bs', type=int, default=15)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'
    classes = list(cat2id) if args.classchoice == 'all' else args.classchoice.split(',')

    model, _ = clip.load('ViT-B/16', device=dev)
    model.eval()
    per_class, all_ious, ms = [], [], []
    for c in classes:
        d = load_class(c, 'ViT-B/16')
        n = min(args.limit, d['pc'].shape[0]) if args.limit else d['pc'].shape[0]
        text = F.normalize(textual_encoder(model, c, device=dev)[0].float(), dim=-1)
        vw = torch.tensor(best_vweight[c], device=dev)
        preds, labs = [], []
        for s in range(0, n, args.bs):
            e = min(s + args.bs, n)
            feat = vanilla_upprojection(d['feat'][s:e].to(dev), d['ifseen'][s:e].to(dev),
                                        d['pointloc'][s:e].to(dev), img_size=(224, 224),
                                        n_points=2048, vweights=vw)[0].float()
            xyz, nrm, fp = (d[q][s:e].to(dev) for q in ('pc', 'normal', 'fpfh'))
            if dev.startswith('cuda'):
                torch.cuda.synchronize()
            t0 = time.time()
            p_ = simple_geoze(xyz, nrm, fp, feat, text, args.n_sp, args.knn,
                              args.rounds, args.part, args.seed)
            if dev.startswith('cuda'):
                torch.cuda.synchronize()
            ms.append((time.time() - t0) * 1e3 / xyz.shape[0])
            preds.append(p_.cpu())
            labs.append(d['label'][s:e])
        pred, lab = torch.cat(preds).numpy(), torch.cat(labs).numpy()
        ious = np.array(calculate_shape_IoU(pred, lab, np.full(pred.shape[0], cat2id[c]),
                                            c, eva=True)[0])
        per_class.append(ious.mean() * 100)
        all_ious.append(ious)
        print(f'  {c:12s} n={pred.shape[0]:4d}  IoU {ious.mean() * 100:6.2f}', flush=True)
    print(f'\nRESULT simple_geoze  class-mIoU={np.mean(per_class):.2f}  '
          f'instance-mIoU={np.concatenate(all_ious).mean() * 100:.2f}  '
          f'{np.mean(ms[1:] if len(ms) > 1 else ms):.2f} ms/shape  '
          f'({args.part}/{args.seed}, n_sp={args.n_sp}, refine={args.rounds}, '
          f'{len(classes)} classes)', flush=True)


if __name__ == '__main__':
    main()
