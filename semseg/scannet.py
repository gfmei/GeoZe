"""ScanNet v2 scene loading for zero-shot semantic segmentation (mirrors partseg/shapenet.py).

Everything is index-aligned to the `vh_clean_2` mesh vertices:

    <s3d_root>/{train,val}/<scene>_vh_clean_2.pth   coord [N,3] f32, color [N,3] in [-1,1], label [N] (255=ignore)
    <feat_root>/<scene>.pth                      {'feat': [V,512] f16, 'mask': [N] bool, 'n': N}
    <scans_root>/<scene>/<scene>_vh_clean_2.0.010000.segs.json   official mesh over-segmentation

`feat` holds the fused 2D VLM (LSeg) features; ~5% of vertices were never observed by a frame
and carry no feature — `fmask` is the validity signal, and those points take their region vector
outright at recovery time.

CAVEAT worth knowing: the mesh over-segmentation is LABEL-PERFECT (assigning each segment its
dominant GT label scores mIoU 1.0000), because ScanNet's annotators labelled these very segments.
So part of the per-point -> per-segment jump is an annotation artifact, not a method gain.  The
VCCS partition (`--part vccs`) has no such leak and is the honest training-free setting.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

S3D_ROOT = '/leonardo_work/IscrC_ERAR/data/scannet/scannet_3d'
# directory name is OpenScene's; the contents are just fused per-point LSeg features
FEAT_ROOT = '/leonardo_work/IscrC_ERAR/data/scannet/scannet_lseg_teacher'
SCANS_ROOT = '/leonardo_work/AIFPT_agrifood/data/scannet/scans'
GEO_CACHE = '/leonardo_scratch/fast/AIFPT_agrifood/cache/geoze_scannet'

IGNORE = 255
NUM_CLASSES = 20


def scene_list(split='val', limit=0, stride=0, s3d_root=S3D_ROOT):
    names = [l.strip() for l in open(os.path.join(s3d_root, f'scannetv2_{split}.txt')) if l.strip()]
    if stride > 1:
        names = names[::stride]
    return names[:limit] if limit else names


def mesh_superpoints(scene, n_points, scans_root=SCANS_ROOT):
    """Official ScanNet mesh over-segmentation (segIndices), compacted to [0,S)."""
    p = os.path.join(scans_root, scene, f'{scene}_vh_clean_2.0.010000.segs.json')
    if not os.path.exists(p):
        return None
    seg = np.asarray(json.load(open(p))['segIndices'], dtype=np.int64)
    if seg.shape[0] != n_points:
        return None
    return np.unique(seg, return_inverse=True)[1].astype(np.int64)


class ScanNetSeg(Dataset):
    """Per-scene tensors for zero-shot evaluation.

    __getitem__ -> dict(scene, xyz, rgb, normal, fpfh, feat, fmask, label, seg)
    `seg` is the VCCS partition (`part='vccs'`) or the mesh over-segmentation (`part='mesh'`).
    """

    def __init__(self, split='val', part='mesh', limit=0, stride=0,
                 s3d_root=S3D_ROOT, feat_root=FEAT_ROOT, scans_root=SCANS_ROOT,
                 geo_cache=GEO_CACHE):
        self.split, self.part = split, part
        self.s3d_root, self.feat_root = s3d_root, feat_root
        self.scans_root, self.geo_cache = scans_root, geo_cache
        self.scenes = scene_list(split, limit, stride, s3d_root)

    def __len__(self):
        return len(self.scenes)

    def geo_path(self, scene):
        return os.path.join(self.geo_cache, f'{scene}_geo.npz')

    def __getitem__(self, i):
        scene = self.scenes[i]
        coord, color, label = torch.load(
            os.path.join(self.s3d_root, self.split, f'{scene}_vh_clean_2.pth'),
            map_location='cpu', weights_only=False)
        N = coord.shape[0]
        t = torch.load(os.path.join(self.feat_root, f'{scene}.pth'),
                       map_location='cpu', weights_only=False)
        fmask = t['mask'].numpy().astype(bool)
        feat = np.zeros((N, t['feat'].shape[1]), np.float32)
        feat[fmask] = t['feat'].float().numpy()

        geo = np.load(self.geo_path(scene))                      # normals + FPFH + VCCS, see sem_prep
        seg = geo['vccs'].astype(np.int64) if self.part == 'vccs' else mesh_superpoints(scene, N)
        if seg is None:
            seg = geo['vccs'].astype(np.int64)
        return {
            'scene': scene,
            'xyz': np.ascontiguousarray(coord.astype(np.float32)),
            'rgb': np.ascontiguousarray(((color.astype(np.float32) + 1.0) * 0.5).clip(0, 1)),
            'normal': geo['normal'].astype(np.float32),
            'fpfh': geo['fpfh'].astype(np.float32),
            'feat': feat,
            'fmask': fmask,
            'label': label.astype(np.int64),
            'seg': np.unique(seg, return_inverse=True)[1].astype(np.int64),
        }
