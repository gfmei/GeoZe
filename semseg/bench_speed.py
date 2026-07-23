"""Inference-time benchmark for the scene aggregation (paper Table 6 comparable).

Table 6 of the GeoZe paper times the AGGREGATION on one scene, not the disk IO or the VLM
backbone.  This measures the same thing: everything from "per-point features are on the GPU"
to "refined per-point features are on the GPU", with CUDA synchronisation around each stage.

    python semseg/bench_speed.py --dataset scannet --limit 50
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semseg.semmodel.best_param import best_param  # noqa: E402
from semseg.semmodel.common import estimate_normals, seg_mean  # noqa: E402
from semseg.semmodel.curve import kdtree_graph, knn_graph  # noqa: E402
from semseg.semmodel.semgeozev2 import SemGeoZeV2  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='scannet', choices=['scannet', 'nuscenes'])
    ap.add_argument('--limit', type=int, default=50)
    ap.add_argument('--warmup', type=int, default=3)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    key = 'scannet20' if args.dataset == 'scannet' else 'nuscenes16'
    p = best_param[key]

    if args.dataset == 'scannet':
        from semseg.scannet import ScanNetSeg
        ds = ScanNetSeg(part='mesh', limit=args.limit)
    else:
        from semseg.nuscenes import NuScenesSeg
        ds = NuScenesSeg(limit=args.limit)

    model = SemGeoZeV2(n_anchor=p['n_anchor'], q_gamma=p['q_gamma'], th_f=p['th_f'], th_n=p['th_n'],
                     rounds=p['rounds'], gamma0=p['gamma0'], w_c=p['w_c']).to(dev).eval()
    T = {k: [] for k in ('graph_kd', 'graph_curve', 'merge_kd', 'merge_curve', 'meanpool')}
    npts, nreg = [], []

    for n in range(len(ds)):
        d = ds[n]
        feat = F.normalize(torch.from_numpy(d['feat']).to(dev), dim=-1)
        valid = torch.from_numpy(d['fmask']).to(dev)
        seg = torch.from_numpy(d['seg']).to(dev)
        xyz = torch.from_numpy(d['xyz']).to(dev)
        rgb = None if d['rgb'] is None else torch.from_numpy(d['rgb']).to(dev)
        fpfh = None if d['fpfh'] is None else torch.from_numpy(d['fpfh']).to(dev)
        S = int(d['seg'].max()) + 1
        warm = n < args.warmup

        sync(); t = time.time()
        z = F.normalize(seg_mean(feat, seg, S, w=valid.to(feat.dtype)), dim=-1)
        sync()
        if not warm:
            T['meanpool'].append(time.time() - t)

        for tag, fn in (('kd', lambda: kdtree_graph(d['xyz'], p['knn'], dev)),
                        ('curve', lambda: knn_graph(xyz, k=p['knn'], min_votes=p['min_votes'],
                                                    window=8, voxel=p['curve_voxel'],
                                                    max_dist=p['curve_max_dist']))):
            sync(); t = time.time()
            pairs = fn()
            sync(); t_graph = time.time() - t
            nrm = (torch.from_numpy(d['normal']).to(dev) if d['normal'] is not None
                   else estimate_normals(xyz, pairs[0], pairs[1]))
            sync(); t = time.time()
            out, _, reg = model(xyz, rgb, nrm, fpfh, feat, valid, seg, pairs)
            sync(); t_merge = time.time() - t
            if not warm:
                T[f'graph_{tag}'].append(t_graph); T[f'merge_{tag}'].append(t_merge)
                if tag == 'kd':
                    nreg.append(int(reg.max().item()) + 1)
        if not warm:
            npts.append(d['xyz'].shape[0])

    print(f'\n=== {args.dataset}: aggregation time, {len(npts)} scenes, '
          f'{np.mean(npts):.0f} points/scene, {np.mean(nreg):.0f} regions ===')
    print(f"  region mean pooling (baseline)      {np.mean(T['meanpool'])*1e3:7.2f} ms")
    for tag, label in (('kd', 'KD-tree graph'), ('curve', 'multi-curve voting graph')):
        g, m = np.mean(T[f'graph_{tag}']) * 1e3, np.mean(T[f'merge_{tag}']) * 1e3
        print(f'  SemGeoZe v2, {label:23s} {g+m:7.2f} ms   (graph {g:6.2f} + aggregation {m:6.2f})')


if __name__ == '__main__':
    main()
