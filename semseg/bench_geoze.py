"""Time the ORIGINAL GeoZe aggregation on ScanNet-scale input, on the same GPU as ours.

The 2125.61 ms in paper Table 6 was measured on the authors' hardware, so the speed-up we quote
is cross-hardware and only indicative. This runs `partseg.partmodel.partgeoze.PartGeoZe` — GeoZe's
own aggregation code, unmodified — on real ScanNet scenes on this machine, so the comparison is
controlled.

Caveat that must travel with the number: the repo ships GeoZe's OBJECT-level configuration, not
the scene configuration that produced Table 6 (that code was never released). `--n_pts` is swept
so the cost/superpoint-count trade-off is visible rather than hidden in one arbitrary setting.
Scenes are subsampled to `--points` because the dense O(N^2)-ish steps do not fit otherwise —
which is itself the finding.

    python semseg/bench_geoze.py --points 20000 40000 --n_pts 128 256
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'partseg'))

from semseg.scannet import ScanNetSeg  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', type=int, default=5)
    ap.add_argument('--points', type=int, nargs='+', default=[10000, 20000, 40000])
    ap.add_argument('--n_pts', type=int, nargs='+', default=[128, 256])
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    from partseg.partmodel.partgeoze import PartGeoZe

    ds = ScanNetSeg(part='mesh', limit=args.scenes)
    scenes = [ds[i] for i in range(len(ds))]
    print(f'GeoZe (partseg/partmodel/partgeoze.py) on {len(scenes)} ScanNet scenes, '
          f'{torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}\n')
    print(f'{"points":>8} {"n_pts":>6} {"time/scene":>12}   note')

    rows = []
    for npt in args.points:
        for nsp in args.n_pts:
            model = PartGeoZe(sigma_d=10.0, sigma_a=0.01, sigma_e=0.001,
                              angle_k=10, n_pts=nsp).to(dev).eval()
            ts, note = [], ''
            for d in scenes:
                N = d['xyz'].shape[0]
                idx = np.linspace(0, N - 1, min(npt, N)).astype(np.int64)
                xyz = torch.from_numpy(d['xyz'][idx]).unsqueeze(0).to(dev)
                nrm = torch.from_numpy(d['normal'][idx]).unsqueeze(0).to(dev)
                fpfh = torch.from_numpy(d['fpfh'][idx]).unsqueeze(0).to(dev)
                feat = torch.from_numpy(d['feat'][idx]).unsqueeze(0).to(dev)
                try:
                    sync(); t = time.time()
                    model(xyz, nrm, fpfh, feat, None)
                    sync(); ts.append(time.time() - t)
                except torch.cuda.OutOfMemoryError:
                    note = 'CUDA OOM'; torch.cuda.empty_cache(); break
                except Exception as e:
                    note = f'{type(e).__name__}: {str(e)[:40]}'; break
            if ts:
                ms = np.mean(ts[1:] or ts) * 1e3
                print(f'{npt:>8} {nsp:>6} {ms:>10.2f} ms   {note}')
                rows.append((npt, nsp, ms))
            else:
                print(f'{npt:>8} {nsp:>6} {"—":>12}   {note}')
            del model; torch.cuda.empty_cache()

    if rows:
        print('\nfor comparison, on this same GPU (semseg/bench_speed.py, 166k points/scene):')
        print('   SemGeoZe v2, multi-curve voting     81.89 ms   (full-resolution scene)')
        best = min(rows, key=lambda r: r[2])
        print(f'\nGeoZe cheapest configuration measured: {best[2]:.2f} ms at {best[0]} points '
              f'/ {best[1]} superpoints — note this is a {166000/best[0]:.0f}x SUBSAMPLED scene, '
              f'so it is a lower bound on the full-resolution cost.')


if __name__ == '__main__':
    main()
