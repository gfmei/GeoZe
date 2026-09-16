"""Stage timing for simple_geoze, so the loops that cost something are visible."""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath('../'))
from partseg.partmodel.eval_v2 import load_class  # noqa: E402
from partseg.simple_geoze import (affinity, cue_features, fps_seed, gather_nb, kmeans,  # noqa: E402
                                  knn, morton_seed, relabel, spectral, split_and_clean)


@torch.no_grad()
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cls', default='chair')
    ap.add_argument('--bs', type=int, default=15)
    ap.add_argument('--n_sp', type=int, default=128)
    ap.add_argument('--reps', type=int, default=5)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = load_class(a.cls, 'ViT-B/16')
    xyz = d['pc'][:a.bs].to(dev)
    nrm = F.normalize(d['normal'][:a.bs].to(dev), dim=-1)
    fp = d['fpfh'][:a.bs].to(dev)
    K = a.n_sp

    def t(name, fn, store):
        torch.cuda.synchronize() if dev == 'cuda' else None
        t0 = time.time()
        out = fn()
        torch.cuda.synchronize() if dev == 'cuda' else None
        store[name] = store.get(name, 0.0) + (time.time() - t0) * 1e3 / a.bs
        return out

    for seed in ('fps', 'curve'):
        for part in ('spectral', 'kmeans'):
            T = {}
            for r in range(a.reps + 1):
                idx = t('knn', lambda: knn(xyz, 10), T)
                X, g, n = t('cues', lambda: cue_features(xyz, nrm, fp, idx), T)
                w = t('affinity (incl. orient)', lambda: affinity(xyz, n, g, idx), T)
                if part == 'spectral':
                    emb = t('spectral', lambda: spectral(w, idx, K), T)
                else:
                    emb = X
                t('seed only', lambda: (morton_seed(xyz, K) if seed == 'curve'
                                        else fps_seed(xyz, K)), T)
                lab = t('kmeans (incl. seed)', lambda: kmeans(emb, xyz, K, seed=seed), T)
                lab = t('relabel', lambda: relabel(lab, w, idx, K, 3), T)
                t('split+clean', lambda: split_and_clean(lab, idx), T)
                if r == 0:
                    T = {}
            tot = sum(v for k, v in T.items() if k != 'seed only')
            print(f'\n=== {part}/{seed}, n_sp={K}   {tot / a.reps:.2f} ms/shape')
            for k, v in sorted(T.items(), key=lambda x: -x[1]):
                mark = '   (inside kmeans)' if k == 'seed only' else ''
                print(f'    {k:24s} {v / a.reps:7.3f} ms/shape{mark}')


if __name__ == '__main__':
    main()
