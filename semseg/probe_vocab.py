"""Vocabulary probe: separate the TEXT TABLE's contribution from the aggregation's.

Runs the label-set / prompt variants against both per-point and region-pooled features, so a
change in mIoU can be attributed to the vocabulary rather than to the aggregation.

OpenScene does not classify nuScenes against the 16 eval names directly: it encodes a 43-entry
DETAILED label set ('barricade', 'bulldozer', 'road', 'curb', 'grass', 'tree trunk', ...) and
maps the argmax back to the 16 classes.  This probe measures both, for per-point and for
region mean pooling, so the aggregation and the text table are separated.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semseg.metrics import ConfusionMatrix
from semseg.nuscenes import NuScenesSeg
from semseg.semmodel.best_param import (NUSCENES16, NUSCENES_DETAILS,
                                        NUSCENES_DETAILS_MAP as MAP_DETAILS)
from semseg.semmodel.common import seg_mean
from semseg.semmodel.post_search import textual_encoder

ENC = 'openai/clip-vit-large-patch14'


@torch.no_grad()
def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    stride = int(os.environ.get('STRIDE', '300'))
    tables = {}
    for name, labels, tmpl in [
        ('16 plain', NUSCENES16, '{}'),
        ('16 "a {} in a scene"', NUSCENES16, 'a {} in a scene'),
        ('43 detail plain', NUSCENES_DETAILS, '{}'),
        ('43 detail "a {} in a scene"', NUSCENES_DETAILS, 'a {} in a scene'),
    ]:
        t = textual_encoder([tmpl], list(labels), ENC).to(dev)
        m = torch.tensor(MAP_DETAILS, device=dev) if len(labels) == 43 else None
        tables[name] = (t, m)

    cms = {f'{k}/{p}': ConfusionMatrix(16) for k in tables for p in ('point', 'meanpool')}
    ds = NuScenesSeg(stride=stride)
    t0 = time.time()
    for n in range(len(ds)):
        d = ds[n]
        feat = F.normalize(torch.from_numpy(d['feat']).to(dev), dim=-1)
        valid = torch.from_numpy(d['fmask']).to(dev)
        seg = torch.from_numpy(d['seg']).to(dev)
        S = int(d['seg'].max()) + 1
        z = F.normalize(seg_mean(feat, seg, S, w=valid.to(feat.dtype)), dim=-1)
        for k, (t, m) in tables.items():
            for p, x in (('point', feat), ('meanpool', z)):
                a = (x @ t.T).argmax(-1)
                a = a if m is None else m[a]
                a = a.cpu().numpy()
                cms[f'{k}/{p}'].add(d['label'], a if p == 'point' else a[d['seg']])

    print(f'\n=== nuScenes text/aggregation probe  {len(ds)} scans  {time.time()-t0:.0f}s ===')
    for k in cms:
        r = cms[k].scores()
        print(f"  {k:34s} mIoU={r['miou']:.4f}  mAcc={r['macc']:.4f}  OA={r['oa']:.4f}")
    best = max(cms, key=lambda k: cms[k].scores()['miou'])
    r = cms[best].scores()
    print(f'\nper-class IoU ({best}):')
    for c, v, pr in zip(NUSCENES16, r['iou'], r['present']):
        if pr:
            print(f'    {c:22s} {v:.3f}')


if __name__ == '__main__':
    main()
