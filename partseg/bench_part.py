"""Where does a shape's time actually go?  Stage-by-stage profile of the v2 partition.

The pooling and the text match are trivially cheap -- classifying 64 superpoint vectors instead
of 2048 point vectors saves real multiply-adds but they were never the cost.  Everything
expensive is in BUILDING the partition, and this prints which part of it, so the optimisation
goes where the milliseconds are.

    python bench_part.py --n_sp 64 --reps 5
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath('../'))

from libs.lib_utils import farthest_point_sample  # noqa: E402
from partseg.partmodel.eval_v2 import load_class  # noqa: E402
from partseg.partmodel.spectral import (absorb_small, compact, connected_components,  # noqa: E402
                                        curve_sample, dense_affinity, edge_affinity, gather_s,
                                        graph_taus, hist_embed, kmeans, knn_idx, nystrom_embedding,
                                        orient_normals, refine_labels, spectral_embedding)
from semseg.semmodel.common import seg_mean  # noqa: E402


class Timer:
    def __init__(self, dev):
        self.dev, self.t = dev, {}

    def __call__(self, name, fn):
        if self.dev.startswith('cuda'):
            torch.cuda.synchronize()
        t0 = time.time()
        out = fn()
        if self.dev.startswith('cuda'):
            torch.cuda.synchronize()
        self.t[name] = self.t.get(name, 0.0) + (time.time() - t0) * 1e3
        return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cls', default='chair')
    ap.add_argument('--bs', type=int, default=15)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--n_sp', type=int, default=64)
    ap.add_argument('--knn', type=int, default=10)
    ap.add_argument('--n_land', type=int, default=256)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'
    d = load_class(args.cls, 'ViT-B/16')
    B = args.bs
    xyz = d['pc'][:B].to(dev)
    nrm = F.normalize(d['normal'][:B].to(dev), dim=-1)
    fp = d['fpfh'][:B].to(dev)
    feat = F.normalize(torch.randn(B, 2048, 512, device=dev), dim=-1)
    text = F.normalize(torch.randn(4, 512, device=dev), dim=-1)
    K = args.n_sp

    for path in ('dense', 'nystrom'):
        for sampler, ortho in ((('fps', True), ('curve', True), ('curve', False))
                               if path == 'nystrom' else (('fps', True),)):
            T = Timer(dev)
            for _ in range(args.reps + 1):
                gfe = T('hist_embed', lambda: hist_embed(fp))
                idx = T('knn_graph', lambda: knn_idx(xyz, args.knn))
                m = T('orient_normals', lambda: orient_normals(xyz, nrm, idx)[0])
                if path == 'dense':
                    W = T('affinity', lambda: dense_affinity(
                        edge_affinity(xyz, nrm, gfe, idx, 1, 1, 1, 1.0, m), idx))
                    emb = T('eigh NxN', lambda: spectral_embedding(W, K))
                else:
                    taus, sig = T('taus', lambda: graph_taus(xyz, m, gfe, idx, True))
                    T('landmarks', lambda: (curve_sample(xyz, args.n_land) if sampler == 'curve'
                                            else farthest_point_sample(xyz, args.n_land, True)))
                    emb = T('nystrom', lambda: nystrom_embedding(
                        (xyz, m, gfe), taus, {'x': 1, 'n': 1, 'g': 1, 'v': 1.0}, K,
                        args.n_land, sig, land=sampler, ortho=ortho))
                lab = T('kmeans', lambda: kmeans(emb, K, 20, seed=sampler))
                lab = T('refine', lambda: refine_labels(
                    lab, edge_affinity(xyz, nrm, gfe, idx, 1, 1, 1, 1.0, m), idx, K, 3))
                seg = T('split+absorb', lambda: absorb_small(
                    compact(lab * 2048 + connected_components(lab, idx)), idx, 4))
                S = int(seg.max()) + 1
                fn = feat.reshape(B * 2048, -1)
                z = T('pool', lambda: F.normalize(seg_mean(fn, seg, S), dim=-1))
                T('classify superpoint', lambda: (z @ text.T).argmax(-1)[seg])
                T('classify per-point', lambda: (fn @ text.T).argmax(-1))
                if _ == 0:
                    T.t = {}                                  # drop the warm-up rep
            tag = f'{path}' + (f'/{sampler}-sample/{"ortho" if ortho else "plain"}'
                               if path == 'nystrom' else '')
            tot = sum(v for k, v in T.t.items() if k != 'classify per-point')
            print(f'\n=== {tag}   {tot / args.reps / B:.2f} ms/shape total')
            for k, v in sorted(T.t.items(), key=lambda x: -x[1]):
                print(f'    {k:22s} {v / args.reps / B:8.3f} ms/shape'
                      f'{"   (not counted)" if k == "classify per-point" else ""}')


if __name__ == '__main__':
    main()
