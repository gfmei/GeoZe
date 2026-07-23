"""Zero-shot semantic segmentation on ScanNet v2 / nuScenes (mirrors partseg/part_run.py).

    python sem_run.py                                   # SemGeoZeV2, ScanNet val (312 scenes)
    python sem_run.py --baseline meanpool               # region mean pooling, the baseline
    python sem_run.py --part vccs                       # self-contained partition, no mesh segments
    python sem_run.py --dataset nuscenes                # nuScenes lidarseg val (6018 scans)
    python sem_run.py --dataset nuscenes --stride 20    # quick subset

Reference numbers (see semseg/README.md):

    ScanNet val, 312 scenes, LSeg, mesh segments
        per-point argmax                mIoU 0.4972   mAcc 0.6346   OA 0.7598
        region mean pooling (baseline)  mIoU 0.5564   mAcc 0.6834   OA 0.8147
        SemGeoZeV2                        mIoU 0.5606   mAcc 0.6856   OA 0.8143
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath('../'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semseg.metrics import ConfusionMatrix  # noqa: E402
from semseg.semmodel.best_param import CLASS_NAMES, LABEL_MAP, best_param  # noqa: E402
from semseg.semmodel.common import estimate_normals, seg_mean  # noqa: E402
from semseg.semmodel.curve import kdtree_graph, knn_graph  # noqa: E402
from semseg.semmodel.post_search import search_prompt  # noqa: E402
from semseg.semmodel.semgeozev2 import SemGeoZeV2  # noqa: E402

warnings.filterwarnings('ignore')

DATASETS = {
    'scannet': dict(key='scannet20', n_classes=20, parts=('mesh', 'vccs'), default_part='mesh'),
    'nuscenes': dict(key='nuscenes16', n_classes=16, parts=('spt',), default_part='spt'),
}


def build_dataset(args, cfg):
    if args.dataset == 'scannet':
        from semseg.scannet import ScanNetSeg
        return ScanNetSeg(split=args.split, part=args.part, limit=args.limit, stride=args.stride)
    from semseg.nuscenes import NuScenesSeg
    return NuScenesSeg(split=args.split, part=args.part, limit=args.limit, stride=args.stride)


@torch.no_grad()
def main(args):
    dcfg = DATASETS[args.dataset]
    args.part = args.part or dcfg['default_part']
    device = args.device if torch.cuda.is_available() else 'cpu'
    p = dict(best_param[dcfg['key']])
    for k in ('th_f', 'th_n', 'rounds', 'knn', 'min_votes', 'gamma0', 'n_anchor'):
        if getattr(args, k, None) is not None:
            p[k] = getattr(args, k)

    text = search_prompt(dcfg['key']).to(device)
    lmap = LABEL_MAP[dcfg['key']]
    lmap = None if lmap is None else torch.tensor(lmap, device=device)

    def classify(x):
        """cosine vs the text table -> EVAL class id (via the detail map where there is one)."""
        a = (x @ text.T).argmax(-1)
        return (a if lmap is None else lmap[a]).cpu().numpy()

    model = SemGeoZeV2(n_anchor=p['n_anchor'], q_gamma=p['q_gamma'], th_f=p['th_f'], th_n=p['th_n'],
                     rounds=p['rounds'], gamma0=p['gamma0'], use_intra=args.intra,
                     alpha_min=args.alpha_min, w_c=p['w_c']).to(device).eval()

    ds = build_dataset(args, dcfg)
    cm = ConfusionMatrix(dcfg['n_classes'])
    nreg, t0 = [], time.time()

    for n in range(len(ds)):
        d = ds[n]
        feat = F.normalize(torch.from_numpy(d['feat']).to(device), dim=-1)
        valid = torch.from_numpy(d['fmask']).to(device)
        seg = torch.from_numpy(d['seg']).to(device)
        S = int(d['seg'].max()) + 1

        if args.baseline == 'point':
            pred = classify(feat)
        elif args.baseline == 'meanpool':
            z = F.normalize(seg_mean(feat, seg, S, w=valid.to(feat.dtype)), dim=-1)
            pred = classify(z)[d['seg']]
            nreg.append(S)
        else:
            xyz = torch.from_numpy(d['xyz']).to(device)
            rgb = None if d['rgb'] is None else torch.from_numpy(d['rgb']).to(device)
            fpfh = None if d['fpfh'] is None else torch.from_numpy(d['fpfh']).to(device)
            pairs = (knn_graph(xyz, k=p['knn'], min_votes=p['min_votes'], window=8,
                               voxel=p['curve_voxel'], max_dist=p['curve_max_dist'])
                     if args.curve else kdtree_graph(d['xyz'], p['knn'], device))
            # LiDAR ships no normals: derive them from the very graph the merge already needs.
            nrm = (torch.from_numpy(d['normal']).to(device) if d['normal'] is not None
                   else estimate_normals(xyz, pairs[0], pairs[1]))
            out, _, reg = model(xyz, rgb, nrm, fpfh, feat, valid, seg, pairs)
            pred = classify(out)
            nreg.append(int(reg.max().item()) + 1)
        cm.add(d['label'], pred)
        if n % max(1, len(ds) // 20) == 0:
            print(f"  [{n+1}/{len(ds)}] {d['scene']} running mIoU={cm.scores()['miou']:.4f} "
                  f'({time.time()-t0:.0f}s)', flush=True)

    r = cm.scores()
    tag = args.baseline or ('semgeozev2' + ('/curve' if args.curve else '') + ('/intra' if args.intra else ''))
    print(f"\nRESULT {args.dataset}/{tag}  part={args.part}  mIoU={r['miou']:.4f}  "
          f"mAcc={r['macc']:.4f}  OA={r['oa']:.4f}  ({len(ds)} scenes, {time.time()-t0:.0f}s"
          f"{f', {np.mean(nreg):.0f} regions' if nreg else ''})", flush=True)
    for c, v, pr in zip(CLASS_NAMES[dcfg['key']], r['iou'], r['present']):
        if pr:
            print(f'    {c:22s} {v:.3f}')
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump({'tag': tag, 'dataset': args.dataset, 'miou': r['miou'], 'macc': r['macc'],
                   'oa': r['oa'], 'iou': r['iou'].tolist(), 'param': p, 'args': vars(args)},
                  open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='scannet', choices=list(DATASETS))
    parser.add_argument('--split', default='val')
    parser.add_argument('--part', default='', help='scannet: mesh|vccs   nuscenes: spt')
    parser.add_argument('--baseline', default='', choices=['', 'point', 'meanpool'])
    parser.add_argument('--curve', action='store_true', help='multi-curve voting graph (no KD-tree)')
    parser.add_argument('--intra', action='store_true', help='enable intra-region consensus attention')
    parser.add_argument('--alpha_min', type=float, default=1.0)
    parser.add_argument('--stride', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--out', default='')
    for k in ('th_f', 'th_n', 'gamma0'):
        parser.add_argument(f'--{k}', type=float, default=None)
    for k in ('rounds', 'knn', 'min_votes', 'n_anchor'):
        parser.add_argument(f'--{k}', type=int, default=None)
    main(parser.parse_args())
