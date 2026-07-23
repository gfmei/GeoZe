"""Bake a handful of ScanNet scenes into the self-contained .npz files the Space loads.

The Space has no ScanNet, no open3d and no GPU, so everything it cannot compute — the geometry
pre-pass (normals, FPFH), the partition, and the fused VLM features — is packed here. Only the
training-free aggregation runs at request time, which is the point of the demo.

    python demo/pack_space_data.py --scenes scene0207_00 scene0583_02 scene0606_01
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semseg.scannet import ScanNetSeg  # noqa: E402
from semseg.semmodel.post_search import search_prompt  # noqa: E402


def voxel_keep(xyz, size):
    _, idx = np.unique(np.floor(xyz / size).astype(np.int64), axis=0, return_index=True)
    return np.sort(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='+', required=True)
    ap.add_argument('--out', default='demo/data')
    ap.add_argument('--voxel', type=float, default=0.03,
                    help='downsample so a CPU Space stays responsive; 0 keeps every point')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    np.save(os.path.join(args.out, 'text.npy'),
            search_prompt('scannet20').numpy().astype(np.float32))

    ds = ScanNetSeg(part='mesh')
    index = {s: i for i, s in enumerate(ds.scenes)}
    for name in args.scenes:
        d = ds[index[name]]
        k = voxel_keep(d['xyz'], args.voxel) if args.voxel > 0 else np.arange(d['xyz'].shape[0])
        fmask = d['fmask'][k]
        seg = np.unique(d['seg'][k], return_inverse=True)[1]
        path = os.path.join(args.out, f'{name}.npz')
        np.savez_compressed(
            path,
            xyz=d['xyz'][k].astype(np.float32), rgb=d['rgb'][k].astype(np.float32),
            normal=d['normal'][k].astype(np.float16), fpfh=d['fpfh'][k].astype(np.float16),
            feat=d['feat'][k][fmask].astype(np.float16), fmask=fmask,
            label=d['label'][k].astype(np.int64), seg=seg.astype(np.int64))
        print(f'{name}: {len(k):>7,} of {d["xyz"].shape[0]:>7,} points, '
              f'{seg.max()+1} segments, {os.path.getsize(path)/1e6:.1f} MB')


if __name__ == '__main__':
    main()
