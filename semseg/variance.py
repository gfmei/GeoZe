"""Is the +0.43 mIoU gain over region mean pooling larger than run-to-run scene variance?

mIoU is computed from ONE pooled confusion matrix over the whole split, so it has no error bar
by construction. The honest way to get one is to resample the unit of independence — the scene.
This records a per-scene confusion matrix for both methods in a single pass, then:

  * bootstrap  — resample the 312 scenes with replacement B times, recompute both mIoUs from the
                 summed confusions, and report the distribution of the DIFFERENCE
  * split-half — 200 random halves, to show how much a 156-scene subset can move the number

A gain is reportable if the bootstrap CI for the difference excludes 0.

    python semseg/variance.py --boot 2000
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semseg.scannet import ScanNetSeg
from semseg.semmodel.best_param import best_param
from semseg.semmodel.common import seg_mean
from semseg.semmodel.curve import kdtree_graph
from semseg.semmodel.post_search import search_prompt
from semseg.semmodel.semgeozev2 import SemGeoZeV2

C = 20


def conf(gt, pred):
    v = (gt >= 0) & (gt < C)
    return np.bincount(gt[v] * C + pred[v], minlength=C * C).reshape(C, C)


def miou(m):
    tp = np.diag(m).astype(np.float64)
    present = m.sum(1) > 0
    iou = tp / np.maximum(m.sum(1) + m.sum(0) - tp, 1)
    return float(iou[present].mean()) if present.any() else 0.0


@torch.no_grad()
def collect(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    text = torch.from_numpy(search_prompt('scannet20')).to(dev) if False else \
        search_prompt('scannet20').to(dev)
    p = best_param['scannet20']
    model = SemGeoZeV2(n_anchor=p['n_anchor'], q_gamma=p['q_gamma'], th_f=p['th_f'],
                       th_n=p['th_n'], rounds=p['rounds'], gamma0=p['gamma0'],
                       w_c=p['w_c']).to(dev).eval()
    ds = ScanNetSeg(part=args.part, stride=args.stride)
    A, B, names = [], [], []
    t0 = time.time()
    for n in range(len(ds)):
        d = ds[n]
        feat = F.normalize(torch.from_numpy(d['feat']).to(dev), dim=-1)
        valid = torch.from_numpy(d['fmask']).to(dev)
        seg = torch.from_numpy(d['seg']).to(dev)
        S = int(d['seg'].max()) + 1
        z = F.normalize(seg_mean(feat, seg, S, w=valid.to(feat.dtype)), dim=-1)
        A.append(conf(d['label'], (z @ text.T).argmax(-1).cpu().numpy()[d['seg']]))
        out, _, reg = model(torch.from_numpy(d['xyz']).to(dev),
                            torch.from_numpy(d['rgb']).to(dev),
                            torch.from_numpy(d['normal']).to(dev),
                            torch.from_numpy(d['fpfh']).to(dev),
                            feat, valid, seg, kdtree_graph(d['xyz'], p['knn'], dev))
        B.append(conf(d['label'], (out @ text.T).argmax(-1).cpu().numpy()))
        names.append(d['scene'])
        if n % 50 == 0:
            print(f'  [{n+1}/{len(ds)}] {d["scene"]} ({time.time()-t0:.0f}s)', flush=True)
    return np.stack(A), np.stack(B), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='mesh')
    ap.add_argument('--stride', type=int, default=0)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--cache', default='semseg/out/per_scene_conf.npz')
    ap.add_argument('--out', default='semseg/out/variance.json')
    args = ap.parse_args()

    if os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        A, B, names = z['a'], z['b'], list(z['names'])
        print(f'loaded {len(names)} per-scene confusions from {args.cache}')
    else:
        A, B, names = collect(args)
        os.makedirs(os.path.dirname(args.cache) or '.', exist_ok=True)
        np.savez_compressed(args.cache, a=A, b=B, names=np.array(names))

    n = len(names)
    base, ours = miou(A.sum(0)), miou(B.sum(0))
    print(f'\nfull split ({n} scenes)   mean pooling {base:.4f}   SemGeoZe v2 {ours:.4f}   '
          f'delta {ours-base:+.4f}')

    rng = np.random.default_rng(0)
    d = np.empty(args.boot)
    ma = np.empty(args.boot); mb = np.empty(args.boot)
    for i in range(args.boot):
        k = rng.integers(0, n, n)
        ma[i] = miou(A[k].sum(0)); mb[i] = miou(B[k].sum(0))
        d[i] = mb[i] - ma[i]
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f'\nbootstrap over scenes (B={args.boot})')
    print(f'  mean pooling   {ma.mean():.4f} +/- {ma.std():.4f}')
    print(f'  SemGeoZe v2    {mb.mean():.4f} +/- {mb.std():.4f}')
    print(f'  delta          {d.mean():+.4f} +/- {d.std():.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]')
    print(f'  P(delta > 0)   {float((d > 0).mean()):.3f}')
    print(f'  -> {"CI excludes 0: the gain is reportable" if lo > 0 else "CI INCLUDES 0: the gain is within scene variance"}')

    h = np.empty(200)
    for i in range(200):
        k = rng.permutation(n)[: n // 2]
        h[i] = miou(B[k].sum(0)) - miou(A[k].sum(0))
    print(f'\nsplit-half ({n//2} scenes, 200 draws)   delta {h.mean():+.4f} +/- {h.std():.4f}   '
          f'range [{h.min():+.4f}, {h.max():+.4f}]')

    json.dump({'n_scenes': n, 'meanpool': base, 'semgeozev2': ours, 'delta': ours - base,
               'boot': {'B': args.boot, 'delta_mean': float(d.mean()), 'delta_std': float(d.std()),
                        'ci95': [float(lo), float(hi)], 'p_gt_0': float((d > 0).mean())},
               'split_half': {'delta_mean': float(h.mean()), 'delta_std': float(h.std()),
                              'min': float(h.min()), 'max': float(h.max())}},
              open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
