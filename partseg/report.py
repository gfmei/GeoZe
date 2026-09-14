"""Collect partseg/out/*.json into the comparison table (mirrors the tables in semseg/README.md)."""
import glob
import json
import os
import sys

import numpy as np

ORDER = ['f_point', 'f_geoze',
         'f_pool_fps_16', 'f_pool_fps_32', 'f_pool_fps_64', 'f_pool_fps_128',
         'f_pool_v2_16', 'f_pool_v2_32', 'f_pool_v2_64', 'f_pool_v2_128',
         'f_oracle_v2_16', 'f_oracle_v2_32', 'f_oracle_v2_64', 'f_oracle_v2_128',
         'point_legacy', 'point', 'geoze', 'meanpool', 'v2', 'oracle']
LABEL = {'f_point': 'per-point argmax', 'f_geoze': 'GeoZe (partgeoze.py)',
         'point_legacy': 'per-point argmax (original layout)', 'point': 'per-point argmax',
         'geoze': 'GeoZe (partgeoze.py)', 'meanpool': 'superpoint mean pooling (baseline)',
         'v2': 'PartGeoZe v2', 'oracle': 'partition oracle (ceiling)'}
for _n in (16, 32, 64, 128):
    LABEL[f'f_pool_fps_{_n}'] = f'  farthest-point superpoints, {_n}'
    LABEL[f'f_pool_v2_{_n}'] = f'  PartGeoZe v2 superpoints, {_n}'
    LABEL[f'f_oracle_v2_{_n}'] = f'  oracle over v2 superpoints, {_n}'


def load(d='out'):
    runs = {}
    for f in glob.glob(os.path.join(d, '*.json')):
        runs[os.path.splitext(os.path.basename(f))[0]] = json.load(open(f))
    return runs


def main():
    runs = load(sys.argv[1] if len(sys.argv) > 1 else 'out')
    if not runs:
        print('no results in out/ yet'); return
    keys = [k for k in ORDER if k in runs] + sorted(set(runs) - set(ORDER))
    cats = sorted({c for r in runs.values() for c in r['classes']})

    print(f'{"":38s} {"cls-mIoU":>9s} {"ins-mIoU":>9s} {"Acc":>7s} {"ms/shape":>9s} {"regions":>8s}')
    for k in keys:
        s, cl = runs[k]['summary'], runs[k]['classes']
        reg = np.mean([v['regions'] for v in cl.values()])
        print(f'{LABEL.get(k, k):38s} {s["class_miou"]:9.2f} {s["instance_miou"]:9.2f} '
              f'{s["acc"]:7.2f} {s["ms"]:9.2f} {reg:8.1f}')

    print(f'\nper-category IoU\n{"category":12s}' + ''.join(f'{k:>16s}' for k in keys))
    for c in cats:
        print(f'{c:12s}' + ''.join(
            f'{runs[k]["classes"][c]["iou"]:>16.2f}' if c in runs[k]['classes'] else f'{"—":>16s}'
            for k in keys))

    pair = next(((x, y) for x, y in (('f_pool_fps_64', 'f_pool_v2_64'), ('meanpool', 'v2'))
                 if x in runs and y in runs), None)
    if pair:
        a = np.concatenate([runs[pair[0]]['shape_ious'][c] for c in cats if c in runs[pair[0]]['shape_ious']])
        b = np.concatenate([runs[pair[1]]['shape_ious'][c] for c in cats if c in runs[pair[1]]['shape_ious']])
        n = min(a.shape[0], b.shape[0])
        d = (b[:n] - a[:n]) * 100
        rng = np.random.default_rng(0)
        boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
        print(f'\n{pair[1]} - {pair[0]}, per shape: {d.mean():+.2f} IoU  '
              f'(bootstrap sd {boot.std():.2f}, 95% CI [{np.percentile(boot, 2.5):+.2f}, '
              f'{np.percentile(boot, 97.5):+.2f}], P(>0) = {(boot > 0).mean():.3f})')


if __name__ == '__main__':
    main()
