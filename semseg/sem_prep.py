"""One-off per-scene geometry cache for ScanNet (normals, FPFH, VCCS partition).

The scene equivalent of what libs.lib_o3d.batch_geo_feature does per shape in partseg — but
ScanNet scenes are ~240k points and the VCCS partition takes a few seconds, so it is computed
once and cached rather than per evaluation run.

    normal  [N,3] f16    open3d PCA normals (r=0.05, k=33)
    fpfh    [N,33] f16   open3d FPFH (r=0.125, k=100), L1-normalised rows
    vccs    [N] i32      VCCS supervoxels from XYZ+RGB+normals only — never the VLM feature, so
                         the partition can not inherit VLM noise
    mesh    [N] i32      official ScanNet mesh over-segmentation (reference partition)

    python semseg/sem_prep.py --shard 0 --nshards 12     # CPU only, embarrassingly parallel
"""
import argparse
import os
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semseg.scannet import GEO_CACHE, S3D_ROOT, mesh_superpoints, scene_list  # noqa: E402
from semseg.semmodel.vccs import VCCS, VccsParams  # noqa: E402


def normals_fpfh(xyz, r_n=0.05, r_f=0.125):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=r_n, max_nn=33))
    fp = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=r_f, max_nn=100))
    nrm = np.asarray(pcd.normals, dtype=np.float32)
    fpfh = np.asarray(fp.data, dtype=np.float32).T
    fpfh /= np.maximum(np.abs(fpfh).sum(1, keepdims=True), 1e-6)      # L1 -> scale-free histogram
    return nrm, fpfh


def vccs_partition(xyz, rgb, nrm, voxel_res=0.02, seed_res=0.25, max_iter=6):
    p = VccsParams(voxel_res=voxel_res, seed_res=seed_res, max_iter=max_iter, seed_mode='grid')
    lab = VCCS(p).fit(xyz.astype(np.float64), rgb.astype(np.float64), nrm.astype(np.float64))[0]
    if (lab < 0).any():                                   # unassigned -> nearest assigned neighbour
        from scipy.spatial import cKDTree
        bad = lab < 0
        lab[bad] = lab[~bad][cKDTree(xyz[~bad]).query(xyz[bad])[1]]
    return np.unique(lab, return_inverse=True)[1].astype(np.int32)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--voxel_res', type=float, default=0.02)
    ap.add_argument('--seed_res', type=float, default=0.25)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    os.makedirs(GEO_CACHE, exist_ok=True)
    scenes = scene_list(args.split)[args.shard::args.nshards]
    t0 = time.time()
    for i, s in enumerate(scenes):
        out = os.path.join(GEO_CACHE, f'{s}_geo.npz')
        if os.path.exists(out) and not args.overwrite:
            continue
        coord, color, _ = torch.load(os.path.join(S3D_ROOT, args.split, f'{s}_vh_clean_2.pth'),
                                     map_location='cpu', weights_only=False)
        xyz = np.ascontiguousarray(coord.astype(np.float32))
        rgb = np.ascontiguousarray(((color.astype(np.float32) + 1.0) * 0.5).clip(0, 1))
        nrm, fpfh = normals_fpfh(xyz)
        vccs = vccs_partition(xyz, rgb, nrm, args.voxel_res, args.seed_res)
        mesh = mesh_superpoints(s, xyz.shape[0])
        np.savez(out, normal=nrm.astype(np.float16), fpfh=fpfh.astype(np.float16), vccs=vccs,
                 mesh=(mesh.astype(np.int32) if mesh is not None else np.zeros(0, np.int32)))
        if i % 10 == 0:
            print(f'[prep {args.shard}] {i+1}/{len(scenes)} {s} S={vccs.max()+1} '
                  f'({time.time()-t0:.0f}s)', flush=True)
    print(f'[prep {args.shard}] done {len(scenes)} scenes in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
