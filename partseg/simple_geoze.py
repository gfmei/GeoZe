"""SimpleGeoZe -- superpoint pooling and label propagation for zero-shot part segmentation.

The method, in four steps and no options:

    1. a geometry-only partition          k-means over position + normals + FPFH, then a few
                                          rounds of affinity-weighted boundary refinement
    2. pool the VLM feature per region    a masked mean of the unit features
    3. classify the REGIONS               S x D against the text table, not N x D
    4. propagate the label to points      exact, since the argmax of a broadcast vector is the
                                          broadcast of its argmax

Nothing here touches the VLM feature until step 2, so a noisy feature can never corrupt the
support it is pooled over.

ShapeNetPart test, all 16 categories, aggregation only on one A100:

    per-point argmax                 50.53 class-mIoU     0.03 ms/shape
    this                             54.82                1.71
    GeoZe (partmodel/partgeoze.py)   56.12               40.30
    partition oracle                 85.91

So this is a SPEED result: 1.30 class-mIoU behind GeoZe at 24x the speed, and 4.3 ahead of
classifying points directly.

A long list of richer designs was measured against this one and none of them won -- hierarchical
merging, intra- and inter-region attention, spectral and cut-pursuit and VCCS partitions, Nystrom
and LOBPCG solvers. partseg/README.md has the numbers, and the git history has the code. The one
finding that explains the list: partition quality is not what limits this task. Sweeping the
superpoint count moves the oracle by 17 class-mIoU and the result by at most 2, and cut pursuit
buys a strictly purer partition and classifies *worse*, because it spends regions where the
geometry varies and leaves too few points in each to pool a stable feature.

    python simple_geoze.py --classchoice all
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]      # repo root, then partseg/ -- runs from anywhere

from common.pointops import EPS, gather_nb, gather_s, knn, seg_mean, split_and_clean  # noqa: E402


# --------------------------------------------------------------------------- #
#  partition                                                                   #
# --------------------------------------------------------------------------- #
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


def fps_seed(xyz, K):
    """Farthest-point seeds, from the point farthest from the centroid so it is deterministic.

    Sequential in K, and that is a measured choice: a Z-order stride is 7x cheaper in seeding but
    costs 0.4 class-mIoU at this K, where the loop is only 32 iterations (+0.16 ms/shape).
    """
    B, N, _ = xyz.shape
    seeds = torch.zeros(B, K, dtype=torch.long, device=xyz.device)
    far = (xyz - xyz.mean(1, keepdim=True)).norm(dim=-1).argmax(1)
    dist = torch.full((B, N), 1e10, device=xyz.device, dtype=xyz.dtype)
    for i in range(K):
        seeds[:, i] = far
        d = (xyz - xyz.gather(1, far.view(B, 1, 1).expand(-1, -1, 3))).pow(2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(1)
    return seeds


def kmeans(X, xyz, K, iters=20):
    """Batched k-means. Every step is one batched op; the loop is over Lloyd rounds only."""
    C = X.gather(1, fps_seed(xyz, K).unsqueeze(-1).expand(-1, -1, X.shape[-1]))
    lab = None
    for _ in range(iters):
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
    problem; sign iteration seeded outward from the centroid settles it in about five rounds and
    reaches 0.84-0.94 edge agreement depending on category, the low end being thin open surfaces
    such as table legs where the orientation is genuinely ambiguous.
    """
    B, N, k = idx.shape
    s = torch.sign((n * (xyz - xyz.mean(1, keepdim=True))).sum(-1))
    s = torch.where(s == 0, torch.ones_like(s), s)
    w = (n.unsqueeze(2) * gather_nb(n, idx)).sum(-1)
    for _ in range(8):
        new = torch.sign((w * gather_s(s, idx)).sum(-1))
        new = torch.where(new == 0, s, new)
        if torch.equal(new, s):
            break
        s = new
    return n * s.unsqueeze(-1)


def affinity(xyz, n, g, idx):
    """Per-edge affinity over four cues, each divided by its own mean so nothing needs tuning.

    The fourth is CONCAVITY: object parts meet at concave seams (seat/leg, wing/body), which is
    the local-convexity criterion, so a concave edge is penalised and a convex one is free.
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


def relabel(lab, w, idx, K, rounds=3):
    """Snap region borders onto the weak edges of the graph.

    k-means places each point independently, so borders come out ragged and cut across strong
    edges.  Each round every point takes the label its strongest neighbours carry.  This is also
    the only route by which the concavity cue reaches the partition, since k-means clusters
    per-point features and concavity is an edge quantity: it moves boundary recall 69.5 -> 72.7
    and the end task by +0.05, so it is about boundaries, not about accuracy.
    """
    B, N, k = idx.shape
    for _ in range(rounds):
        sc = torch.zeros(B, N, K, device=lab.device, dtype=w.dtype)
        sc.scatter_add_(2, gather_s(lab, idx), w)
        new = sc.argmax(-1)
        if torch.equal(new, lab):
            break
        lab = new
    return lab


def superpoints(xyz, nrm, fpfh, n_sp=32, k=10, rounds=3):
    """[B,N] coordinates + normals + FPFH -> flat [B*N] region ids, grouped by shape.

    `n_sp=32` is measured on the END TASK, not on the partition: oracle IoU keeps climbing with
    resolution (69.9 at 16 regions -> 87.2 at 128) while class-mIoU peaks at 32 and then falls,
    because smaller regions average fewer features and that variance costs more than the raised
    ceiling gains.  Do not tune this on oracle IoU.
    """
    idx = knn(xyz, k)
    X, g, n = cue_features(xyz, nrm, fpfh, idx)
    lab = kmeans(X, xyz, n_sp)
    if rounds:
        lab = relabel(lab, affinity(xyz, n, g, idx), idx, n_sp, rounds)
    return split_and_clean(lab, idx)


# --------------------------------------------------------------------------- #
#  pool, classify, propagate                                                   #
# --------------------------------------------------------------------------- #
def simple_geoze(xyz, nrm, fpfh, feat, text, n_sp=32, k=10, rounds=3):
    """Per-point part labels [B,N].  `feat` [B,N,D] VLM features, `text` [C,D] L2-normalised."""
    B, N, D = feat.shape
    seg = superpoints(xyz, nrm, fpfh, n_sp, k, rounds)

    f = feat.reshape(B * N, D)
    valid = (f.norm(dim=-1) > 0).to(f.dtype)               # unseen points carry no feature
    z = F.normalize(seg_mean(F.normalize(f, dim=-1), seg, int(seg.max()) + 1, w=valid), dim=-1)

    return (z @ text.t()).argmax(-1)[seg].view(B, N)       # classify regions, propagate labels


# --------------------------------------------------------------------------- #
#  evaluation                                                                  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def main():
    from libs.lib_metric import calculate_shape_IoU, seg_num
    from partseg.partclip import clip
    from partseg.partmodel.best_param import best_vweight
    from partseg.partmodel.post_search import cat2id, textual_encoder
    from partseg.rendering.unprojection import vanilla_upprojection
    from partseg.shapenet import load_class

    ap = argparse.ArgumentParser()
    ap.add_argument('--classchoice', default='all', help='a category, a comma list, or all')
    ap.add_argument('--modelname', default='ViT-B/16')
    ap.add_argument('--n_sp', type=int, default=32)
    ap.add_argument('--knn', type=int, default=10)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--bs', type=int, default=15)
    ap.add_argument('--out', default='', help='json with per-category results')
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'
    classes = list(cat2id) if args.classchoice == 'all' else args.classchoice.split(',')

    model, _ = clip.load(args.modelname, device=dev)
    model.eval()
    per_class, all_ious, ms, by_class = [], [], [], {}
    for c in classes:
        d = load_class(c, args.modelname)
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
            p = simple_geoze(xyz, nrm, fp, feat, text, args.n_sp, args.knn, args.rounds)
            if dev.startswith('cuda'):
                torch.cuda.synchronize()
            ms.append((time.time() - t0) * 1e3 / xyz.shape[0])
            preds.append(p.cpu())
            labs.append(d['label'][s:e])
        pred, lab = torch.cat(preds).numpy(), torch.cat(labs).numpy()
        ious = np.array(calculate_shape_IoU(pred, lab, np.full(pred.shape[0], cat2id[c]),
                                            c, eva=True)[0])
        per_class.append(ious.mean() * 100)
        all_ious.append(ious)
        by_class[c] = float(ious.mean() * 100)
        print(f'  {c:12s} n={pred.shape[0]:4d}  IoU {ious.mean() * 100:6.2f}', flush=True)

    summary = dict(class_miou=float(np.mean(per_class)),
                   instance_miou=float(np.concatenate(all_ious).mean() * 100),
                   ms=float(np.mean(ms[1:] if len(ms) > 1 else ms)))
    print(f"\nRESULT simple_geoze  class-mIoU={summary['class_miou']:.2f}  "
          f"instance-mIoU={summary['instance_miou']:.2f}  {summary['ms']:.2f} ms/shape  "
          f'(n_sp={args.n_sp}, refine={args.rounds}, {len(classes)} classes)', flush=True)
    if args.out:
        import json
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump({**summary, 'by_class': by_class, 'args': vars(args)}, open(args.out, 'w'),
                  indent=1)


if __name__ == '__main__':
    main()
