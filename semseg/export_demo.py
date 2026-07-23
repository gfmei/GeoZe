"""Export a representative set of ScanNet scenes + all prediction variants for the web demos.

Scenes are NOT cherry-picked: candidates are scored, then one is taken from each of the
neutral / typical / good bands of the SemGeoZeV2-minus-mean-pooling distribution, and every
scene ships its own honest per-scene mIoU so the demo cannot imply a gain it did not get.

Writes docs/scenes/<scene>.json (base64 typed arrays) plus docs/scenes/manifest.json:

    xyz    int16 [N,3]  quantised to the scene bbox (scale in the header)
    rgb    uint8 [N,3]
    gt / point / meanpool / semgeozev2   uint8 [N]   class ids, 255 = unlabelled
    region uint16 [N]  merged-region id, so the demo can show the partition itself

    python semseg/export_demo.py --candidates 40
"""
import argparse
import base64
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semseg.metrics import ConfusionMatrix
from semseg.scannet import ScanNetSeg
from semseg.semmodel.best_param import SCANNET20, best_param
from semseg.semmodel.common import seg_mean
from semseg.semmodel.curve import kdtree_graph
from semseg.semmodel.post_search import search_prompt
from semseg.semmodel.semgeozev2 import SemGeoZeV2

# ScanNet's own 20-class colormap — the domain's native palette.
SCANNET_COLORS = [
    (174, 199, 232), (152, 223, 138), (31, 119, 180), (255, 187, 120), (188, 189, 34),
    (140, 86, 75), (255, 152, 150), (214, 39, 40), (197, 176, 213), (148, 103, 189),
    (196, 156, 148), (23, 190, 207), (247, 182, 210), (219, 219, 141), (255, 127, 14),
    (158, 218, 229), (44, 160, 44), (112, 128, 144), (227, 119, 194), (82, 84, 163),
]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode('ascii')


def voxel_downsample(xyz, size):
    key = np.floor(xyz / size).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return np.sort(idx)


@torch.no_grad()
def run_scene(d, model, text, device):
    feat = F.normalize(torch.from_numpy(d['feat']).to(device), dim=-1)
    valid = torch.from_numpy(d['fmask']).to(device)
    seg = torch.from_numpy(d['seg']).to(device)
    S = int(d['seg'].max()) + 1
    xyz = torch.from_numpy(d['xyz']).to(device)
    rgb = torch.from_numpy(d['rgb']).to(device)
    nrm = torch.from_numpy(d['normal']).to(device)
    fpfh = torch.from_numpy(d['fpfh']).to(device)
    pairs = kdtree_graph(d['xyz'], best_param['scannet20']['knn'], device)

    pt = (feat @ text.T).argmax(-1).cpu().numpy()
    z = F.normalize(seg_mean(feat, seg, S, w=valid.to(feat.dtype)), dim=-1)
    mp = (z @ text.T).argmax(-1).cpu().numpy()[d['seg']]
    out, _, reg = model(xyz, rgb, nrm, fpfh, feat, valid, seg, pairs)
    sg = (out @ text.T).argmax(-1).cpu().numpy()
    return pt, mp, sg, reg.cpu().numpy(), S


def export(d, res, outdir, voxel):
    pt, mp, sg, reg, S, sc = res['pt'], res['mp'], res['sg'], res['reg'], res['S'], res['sc']
    keep = voxel_downsample(d['xyz'], voxel)
    xyz = d['xyz'][keep]
    lo, hi = xyz.min(0), xyz.max(0)
    scale = float((hi - lo).max() / 32000.0)
    q = np.round((xyz - (lo + hi) / 2) / scale).astype(np.int16)
    _, rg = np.unique(reg[keep], return_inverse=True)
    lab = d['label'][keep]
    payload = {
        'scene': d['scene'],
        'n_points': int(len(keep)), 'n_points_full': int(d['xyz'].shape[0]),
        'n_regions_in': int(S), 'n_regions_out': int(reg.max() + 1),
        'scale': scale, 'classes': SCANNET20, 'colors': SCANNET_COLORS,
        'miou': {k: float(v) for k, v in sc.items()},
        'xyz': b64(q), 'rgb': b64((d['rgb'][keep] * 255).astype(np.uint8)),
        'gt': b64(np.where((lab >= 0) & (lab < 20), lab, 255).astype(np.uint8)),
        'point': b64(pt[keep].astype(np.uint8)),
        'meanpool': b64(mp[keep].astype(np.uint8)),
        'semgeozev2': b64(sg[keep].astype(np.uint8)),
        'region': b64(rg.astype(np.uint16)),
    }
    path = os.path.join(outdir, f"{d['scene']}.json")
    json.dump(payload, open(path, 'w'))
    return path, len(keep), os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='docs/scenes')
    ap.add_argument('--candidates', type=int, default=40)
    ap.add_argument('--voxel', type=float, default=0.035)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text = search_prompt('scannet20').to(device)
    p = best_param['scannet20']
    model = SemGeoZeV2(n_anchor=p['n_anchor'], q_gamma=p['q_gamma'], th_f=p['th_f'],
                     th_n=p['th_n'], rounds=p['rounds'], gamma0=p['gamma0'],
                     w_c=p['w_c']).to(device).eval()

    ds = ScanNetSeg(part='mesh')
    order = list(range(0, len(ds), max(1, len(ds) // args.candidates)))
    cand = []
    for n in order:
        d = ds[n]
        pt, mp, sg, reg, S = run_scene(d, model, text, device)
        gt = d['label']
        sc = {}
        for k, pr in (('point', pt), ('meanpool', mp), ('semgeozev2', sg)):
            cm = ConfusionMatrix(20); cm.add(gt, pr); sc[k] = cm.scores()['miou']
        ncls = int(len(np.unique(gt[(gt >= 0) & (gt < 20)])))
        cand.append(dict(n=n, scene=d['scene'], ncls=ncls, sc=sc,
                         gain=sc['semgeozev2'] - sc['meanpool'],
                         res=dict(pt=pt, mp=mp, sg=sg, reg=reg, S=S, sc=sc)))
        print(f"  {d['scene']:16s} cls={ncls:2d} point={sc['point']:.3f} "
              f"mean={sc['meanpool']:.3f} ours={sc['semgeozev2']:.3f} ({cand[-1]['gain']:+.3f})",
              flush=True)

    rich = sorted([c for c in cand if c['ncls'] >= 9], key=lambda c: c['gain'])
    if len(rich) < 3:
        rich = sorted(cand, key=lambda c: c['gain'])
    picks = [('neutral', rich[len(rich) // 6]),
             ('typical', rich[len(rich) // 2]),
             ('good', rich[-1 - len(rich) // 8])]

    os.makedirs(args.outdir, exist_ok=True)
    manifest = []
    for band, c in picks:
        d = ds[c['n']]
        path, npts, nbytes = export(d, c['res'], args.outdir, args.voxel)
        manifest.append(dict(scene=c['scene'], band=band, n_classes=c['ncls'],
                             miou={k: round(float(v), 4) for k, v in c['sc'].items()},
                             gain=round(float(c['gain']), 4), points=npts,
                             file=os.path.basename(path)))
        print(f"[{band:8s}] {c['scene']} -> {path}  {npts} pts  {nbytes/1e6:.2f} MB  "
              f"gain {c['gain']:+.4f}")
    json.dump({'scenes': manifest, 'note': 'scenes span the neutral/typical/good bands of the '
                                           'per-scene SemGeoZeV2-minus-meanpool distribution'},
              open(os.path.join(args.outdir, 'manifest.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
