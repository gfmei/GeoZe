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


def kmeans(X, xyz, K, iters=20):
    """Batched k-means, seeded by farthest-point sampling on the coordinates (deterministic)."""
    B, N, _ = X.shape
    seeds = torch.zeros(B, K, dtype=torch.long, device=X.device)
    far = (xyz - xyz.mean(1, keepdim=True)).norm(dim=-1).argmax(1)
    dist = torch.full((B, N), 1e10, device=X.device, dtype=xyz.dtype)
    for i in range(K):
        seeds[:, i] = far
        d = (xyz - xyz.gather(1, far.view(B, 1, 1).expand(-1, -1, 3))).pow(2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(1)
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


def refine(lab, xyz, n, g, idx, K, rounds=3):
    """Snap region borders onto the weak edges of the graph.

    k-means places each point independently, so borders come out ragged and cut across strong
    edges.  Each round every point takes the label its strongest neighbours carry, under an
    affinity that includes CONCAVITY: object parts meet at concave seams (seat/leg, wing/body),
    so a concave edge is penalised and a convex one is free.  Concavity needs consistently
    oriented normals, which an estimator does not give, so the signs are resolved first by
    relaxing max_s sum_ij s_i s_j (n_i.n_j) over the graph, seeded outward from the centroid.
    """
    B, N, k = idx.shape
    s = torch.sign((n * (xyz - xyz.mean(1, keepdim=True))).sum(-1))
    s = torch.where(s == 0, torch.ones_like(s), s)
    w_sign = (n.unsqueeze(2) * gather_nb(n, idx)).sum(-1)
    for _ in range(20):
        new = torch.sign((w_sign * s.gather(1, idx.reshape(B, -1)).reshape(B, N, k)).sum(-1))
        new = torch.where(new == 0, s, new)
        if torch.equal(new, s):
            break
        s = new
    m = n * s.unsqueeze(-1)

    dx = xyz.unsqueeze(2) - gather_nb(xyz, idx)
    d = {'x': dx.pow(2).sum(-1),
         'n': (1 - (n.unsqueeze(2) * gather_nb(n, idx)).sum(-1).abs()).clamp_min(0),
         'g': (1 - (g.unsqueeze(2) * gather_nb(g, idx)).sum(-1)).clamp_min(0),
         'v': F.relu(-((m.unsqueeze(2) - gather_nb(m, idx))
                       * F.normalize(dx, dim=-1)).sum(-1))}
    e = torch.zeros_like(d['x'])
    for v in d.values():
        e = e - v / v.mean(dim=(1, 2), keepdim=True).clamp_min(EPS)
    w = torch.exp(e)
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
    for _ in range(N):
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


def superpoints(xyz, nrm, fpfh, n_sp=32, k=10, rounds=3):
    """[B,N] coordinates + normals + FPFH -> flat [B*N] region ids, grouped by shape."""
    idx = knn(xyz, k)
    X, g, n = cue_features(xyz, nrm, fpfh, idx)
    lab = kmeans(X, xyz, n_sp)
    if rounds:
        lab = refine(lab, xyz, n, g, idx, n_sp, rounds)
    return split_and_clean(lab, idx)


# --------------------------------------------------------------------------- #
#  pool, classify, propagate                                                   #
# --------------------------------------------------------------------------- #
def simple_geoze(xyz, nrm, fpfh, feat, text, n_sp=32, k=10, rounds=3):
    """Per-point part labels [B,N].  `feat` [B,N,D] VLM features, `text` [C,D] L2-normalised."""
    B, N, D = feat.shape
    seg = superpoints(xyz, nrm, fpfh, n_sp, k, rounds)
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
            p_ = simple_geoze(xyz, nrm, fp, feat, text, args.n_sp, args.knn, args.rounds)
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
          f'(n_sp={args.n_sp}, refine={args.rounds}, {len(classes)} classes)', flush=True)


if __name__ == '__main__':
    main()
