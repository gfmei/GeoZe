"""nuScenes lidarseg loading for zero-shot semantic segmentation (mirrors semseg/scannet.py).

OpenScene-native layout, everything index-aligned to the aggregated LiDAR scan:

    <nusc_root>/val/<scene>.pth         tuple (coords [N,3] f64, 0, labels [N] f64, 255=ignore)
    <nusc_root>/val/<scene>_spt.npy     VCCS superpoint ids [N] i32 (1024 per scan, precomputed)
    <feat_root>/<scene>.pt              {"feat": [V,768] f16, "mask_full": [N] bool}

Three things differ from ScanNet and drive the whole configuration:

1. NO COLOUR and NO NORMALS.  `rgb` is returned as zeros and the colour cues are switched off
   (`w_c=0`, and the colour term of the boundary gate is disabled).  Normals are estimated on
   GPU from the neighbour graph the merge already builds — see semmodel/common.estimate_normals
   — so there is no open3d pre-pass for this dataset.
2. OUTDOOR METRIC SCALE (~200 m across, ~0.5 m point spacing at range) rather than a 5 m room,
   so every distance threshold is an order of magnitude larger than the ScanNet defaults.
3. EXTREME FEATURE SPARSITY.  The fused OpenSeg features cover only 5.6% of points — but 75.9%
   of the *labelled* ones (7.3% of points carry a label at all, since multi-sweep aggregation
   labels only the keyframe).  Region pooling is what fills the rest, which is why the pooled
   baseline is so far ahead of per-point classification here.

Features are OpenSeg (768-d), so the text table must come from CLIP ViT-L/14, not the ViT-B/32
tower ScanNet's LSeg features live in.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

NUSC_ROOT = '/leonardo_work/IscrC_ERAR/data/nuscenes/nuscenes_3d'
FEAT_ROOT = '/leonardo_work/IscrC_ERAR/data/nuscenes/nuscenes_multiview_openseg_val'

IGNORE = 255
NUM_CLASSES = 16


def scene_list(split='val', limit=0, stride=0, feat_root=FEAT_ROOT):
    """Scans that actually have fused features (the released split is val-only)."""
    names = sorted(f[:-3] for f in os.listdir(feat_root) if f.endswith('.pt'))
    if stride > 1:
        names = names[::stride]
    return names[:limit] if limit else names


class NuScenesSeg(Dataset):
    """__getitem__ -> dict(scene, xyz, rgb, normal, fpfh, feat, fmask, label, seg).

    `normal` and `fpfh` are None here: normals are derived on GPU from the neighbour graph in
    sem_run, and FPFH is only needed by the optional intra/inter stages (see semgeozev2).
    """

    def __init__(self, split='val', part='spt', limit=0, stride=0,
                 nusc_root=NUSC_ROOT, feat_root=FEAT_ROOT):
        self.split, self.part = split, part
        self.nusc_root, self.feat_root = nusc_root, feat_root
        self.scenes = scene_list(split, limit, stride, feat_root)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        scene = self.scenes[i]
        coord, _, label = torch.load(os.path.join(self.nusc_root, self.split, f'{scene}.pth'),
                                     map_location='cpu', weights_only=False)
        N = coord.shape[0]
        t = torch.load(os.path.join(self.feat_root, f'{scene}.pt'),
                       map_location='cpu', weights_only=False)
        fmask = np.asarray(t['mask_full']).astype(bool)
        feat = np.zeros((N, t['feat'].shape[1]), np.float32)
        feat[fmask] = t['feat'].float().numpy()
        seg = np.load(os.path.join(self.nusc_root, self.split, f'{scene}_spt.npy')).astype(np.int64)
        return {
            'scene': scene,
            'xyz': np.ascontiguousarray(coord.astype(np.float32)),
            'rgb': None,                                   # nuScenes LiDAR has no colour
            'normal': None,                                # derived on GPU from the kNN graph
            'fpfh': None,                                  # only the optional stages need it
            'feat': feat,
            'fmask': fmask,
            'label': label.astype(np.int64),
            'seg': np.unique(seg, return_inverse=True)[1].astype(np.int64),
        }
