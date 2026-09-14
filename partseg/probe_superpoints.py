"""Superpoint QUALITY sweep: how good is the partition, before any feature ever touches it.

Partition quality bounds everything downstream, so it is measured on its own, with the metrics
the superpoint literature uses rather than with the end task:

    oracle IoU   every region takes its majority ground-truth part; the ceiling for any method
                 that pools inside regions
    oracle Acc   the same, as point accuracy = 1 - undersegmentation error (the leakage rate)
    BR / BP      boundary recall and precision on the kNN graph: a point is a boundary point if
                 any neighbour carries a different part (ground truth) or a different region
                 (partition).  BR says whether real part seams are cut; BP says whether the cuts
                 land anywhere real.

All of it is reported against the REGION COUNT, because any partition looks better when it is
finer -- comparisons are only meaningful at a matched number of regions.

    python probe_superpoints.py --classes airplane chair table lamp motorbike --limit 40
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath('../'))

from libs.lib_metric import seg_num  # noqa: E402
from partseg.partmodel.eval_v2 import PC_NUM, load_class, oracle_pred  # noqa: E402
from partseg.partmodel.post_search import cat2id  # noqa: E402
from partseg.partmodel.spectral import (curve_knn, gather_s, hist_embed, knn_idx,  # noqa: E402
                                        orient_normals, superpoints)

# name -> the knobs of partmodel.spectral.superpoints that define this partition
VARIANTS = {
    'fps':            dict(method='fps'),
    'kmeans':         dict(method='kmeans'),
    'spectral':       dict(method='spectral'),
    'spec+cvx':       dict(method='spectral', w_v=1.0),
    'spec+cvx+st':    dict(method='spectral', w_v=1.0, self_tune=True),
    'spec+cvx+st+ev': dict(method='spectral', w_v=1.0, self_tune=True, n_ev=32),
    'spec+cvx2':      dict(method='spectral', w_v=2.0),
    'spec+cvx-only':  dict(method='spectral', w_v=2.0, w_g=0.0),
    'nys':            dict(method='spectral', embed='nystrom'),
    'nys+cvx':        dict(method='spectral', embed='nystrom', w_v=1.0),
    'nys+cvx+st':     dict(method='spectral', embed='nystrom', w_v=1.0, self_tune=True),
    'nys+cvx@512':    dict(method='spectral', embed='nystrom', w_v=1.0, n_land=512),
    'best@64':        dict(method='spectral', embed='nystrom', w_v=1.0, self_tune=True, n_land=64),
    'best@128':       dict(method='spectral', embed='nystrom', w_v=1.0, self_tune=True, n_land=128),
    'best@256':       dict(method='spectral', embed='nystrom', w_v=1.0, self_tune=True, n_land=256),
    'best+refine':    dict(method='spectral', embed='nystrom', w_v=1.0, self_tune=True, n_land=256,
                           refine=3),
    'best+ortho':     dict(method='spectral', embed='nystrom', w_v=1.0, self_tune=True, n_land=256,
                           ortho=True),
    'ortho@64':       dict(method='spectral', embed='nystrom', w_v=1.0, self_tune=True, n_land=64,
                           ortho=True),
    'ortho@128':      dict(method='spectral', embed='nystrom', w_v=1.0, self_tune=True, n_land=128,
                           ortho=True),
    'ortho@128-cvx':  dict(method='spectral', embed='nystrom', w_v=0.0, self_tune=True, n_land=128,
                           ortho=True),
    'dense+cvx+st':   dict(method='spectral', w_v=1.0, self_tune=True),
    'km+ref':         dict(method='kmeans', refine=3, w_v=1.0, self_tune=True),
    'dense':          dict(method='spectral', embed='dense', w_v=1.0, self_tune=True),
    'lobpcg':         dict(method='spectral', embed='lobpcg', w_v=1.0, self_tune=True),
    'sparse30':       dict(method='spectral', embed='sparse', w_v=1.0, self_tune=True),
    'sparse100':      dict(method='spectral', embed='sparse', w_v=1.0, self_tune=True,
                           sparse_iters=100),
    'sparse50':       dict(method='spectral', embed='sparse', w_v=1.0, self_tune=True,
                           sparse_iters=50),
    'sparse30+ref':   dict(method='spectral', embed='sparse', w_v=1.0, self_tune=True, refine=3),
    'dense/curveknn': dict(method='spectral', embed='dense', w_v=1.0, self_tune=True,
                           curve_knn_window=32),
    'lobpcg/curveknn': dict(method='spectral', embed='lobpcg', w_v=1.0, self_tune=True,
                            curve_knn_window=32),
    'km+ref6':        dict(method='kmeans', refine=6, w_v=1.0, self_tune=True),
    'km+ref-nocvx':   dict(method='kmeans', refine=3, w_v=0.0, self_tune=True),
    'fps+ref':        dict(method='fps', refine=3, w_v=1.0, self_tune=True),
}


def boundary(lab, idx):
    """[B*N] labels -> [B*N] bool, true where a kNN neighbour carries a different label."""
    B, N, k = idx.shape
    nb = (idx + (torch.arange(B, device=idx.device) * N).view(B, 1, 1)).reshape(B * N, k)
    return (lab[nb] != lab.unsqueeze(1)).any(1)


def quality(seg, lab, idx, n_parts):
    """oracle IoU (per-shape mean), oracle Acc, boundary recall, boundary precision."""
    B, N, _ = idx.shape
    pred = oracle_pred(seg, lab, n_parts)
    acc = float((pred == lab).float().mean())
    p, g = pred.view(B, N), lab.view(B, N)
    ious = []
    for b in range(B):
        pi = []
        for c in range(n_parts):
            inter = ((p[b] == c) & (g[b] == c)).sum().item()
            union = ((p[b] == c) | (g[b] == c)).sum().item()
            pi.append(1.0 if union == 0 else inter / union)
        ious.append(np.mean(pi))
    gb, sb = boundary(lab, idx), boundary(seg, idx)
    br = float((gb & sb).sum() / gb.sum().clamp_min(1))
    bp = float((gb & sb).sum() / sb.sum().clamp_min(1))
    return np.mean(ious) * 100, acc * 100, br * 100, bp * 100


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', nargs='+',
                    default=['airplane', 'chair', 'table', 'lamp', 'motorbike', 'car'])
    ap.add_argument('--limit', type=int, default=40)
    ap.add_argument('--n_sp', type=int, nargs='+', default=[32, 64, 128])
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS))
    ap.add_argument('--knn', type=int, default=10)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--bs', type=int, default=10)
    ap.add_argument('--out', default='out/superpoints.json')
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'
    acc = {}

    for c in args.classes:
        d = load_class(c, 'ViT-B/16')
        n = min(args.limit, d['pc'].shape[0]) if args.limit else d['pc'].shape[0]
        P = seg_num[cat2id[c]]
        print(f'\n===== {c}  ({n} shapes, {P} parts)', flush=True)
        print(f'{"variant":16s}{"n_sp":>6s}{"regions":>9s}{"oracleIoU":>11s}{"oracleAcc":>11s}'
              f'{"BR":>8s}{"BP":>8s}{"ms/shape":>10s}')
        agree = []
        for name in args.variants:
            for k in args.n_sp:
                rows, ms = [], []
                for s in range(0, n, args.bs):
                    xyz, nrm, fp = (d[q][s:s + args.bs].to(dev) for q in ('pc', 'normal', 'fpfh'))
                    lab = d['label'][s:s + args.bs].to(dev).reshape(-1)
                    B = xyz.shape[0]
                    nrm = F.normalize(nrm, dim=-1)
                    kw = dict(VARIANTS[name])
                    cw = kw.pop('curve_knn_window', 0)
                    idx = curve_knn(xyz, args.knn, window=cw) if cw else knn_idx(xyz, args.knn)
                    if not agree:
                        agree.append(orient_normals(xyz, nrm, idx)[1])
                    if dev.startswith('cuda'):
                        torch.cuda.synchronize()
                    t0 = time.time()
                    seg = superpoints(xyz, nrm, hist_embed(fp), idx, n_sp=k, **kw)
                    if dev.startswith('cuda'):
                        torch.cuda.synchronize()
                    ms.append((time.time() - t0) * 1e3 / B)
                    rows.append(((int(seg.max()) + 1) / B,) + quality(seg, lab, idx, P))
                r = np.append(np.mean(np.array(rows), axis=0),
                              float(np.mean(ms[1:] if len(ms) > 1 else ms)))
                acc.setdefault(name, {}).setdefault(str(k), []).append(r.tolist())
                print(f'{name:16s}{k:>6d}{r[0]:>9.1f}{r[1]:>11.2f}{r[2]:>11.2f}'
                      f'{r[3]:>8.2f}{r[4]:>8.2f}{r[5]:>10.2f}', flush=True)
        print(f'  (normal-orientation agreement on this class: {agree[0]:.3f})')

    print('\n===== mean over classes')
    print(f'{"variant":16s}{"n_sp":>6s}{"regions":>9s}{"oracleIoU":>11s}{"oracleAcc":>11s}'
          f'{"BR":>8s}{"BP":>8s}{"ms/shape":>10s}')
    summary = {}
    for name, per_k in acc.items():
        for k, rows in per_k.items():
            r = np.mean(np.array(rows), axis=0)
            summary[f'{name}@{k}'] = r.tolist()
            print(f'{name:16s}{int(k):>6d}{r[0]:>9.1f}{r[1]:>11.2f}{r[2]:>11.2f}'
                  f'{r[3]:>8.2f}{r[4]:>8.2f}{r[5]:>10.2f}')
    if args.out:
        import json
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump({'summary': summary, 'per_class': acc, 'args': vars(args)},
                  open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
