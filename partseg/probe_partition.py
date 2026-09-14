"""Partition and merge-criterion probe for PartGeoZe v2 (mirrors semseg/probe_vocab.py).

Two questions decide the design, so measure them instead of guessing:

  1. How good is the partition?   oracle IoU (every superpoint takes its majority GT part) for
     spectral / kmeans / fps at several superpoint counts.  This is the ceiling any pooling over
     that partition can reach.
  2. Can the merge tell parts apart?   for adjacent superpoint pairs, the cosine between the two
     region features — raw and per-shape mean-centred — split by whether the two regions carry
     the same GT part, plus the AUC of each score and the boundary gate b_mn.  The threshold
     th_f is read off these distributions.

    python probe_partition.py --classes airplane chair table lamp --limit 60
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath('../'))

from libs.lib_metric import seg_num  # noqa: E402
from partseg.partmodel.best_param import best_param_v2, best_vweight  # noqa: E402
from partseg.partmodel.eval_v2 import PC_NUM, load_class, oracle_pred  # noqa: E402
from partseg.partmodel.partgeozev2 import PartGeoZeV2  # noqa: E402
from partseg.partmodel.post_search import cat2id  # noqa: E402
from partseg.partmodel.spectral import flat_pairs, hist_embed  # noqa: E402
from partseg.rendering.unprojection import vanilla_upprojection  # noqa: E402
from semseg.semmodel.common import seg_mean  # noqa: E402


def auc(score, pos):
    """Rank AUC of `score` for the binary mask `pos` (ties ignored)."""
    r = torch.empty_like(score)
    r[torch.argsort(score)] = torch.arange(score.numel(), device=score.device, dtype=score.dtype) + 1
    npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        return float('nan')
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def stat(v, m):
    return f'{v[m].mean():.3f}±{v[m].std():.3f}'


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', nargs='+', default=['airplane', 'chair', 'table', 'lamp'])
    ap.add_argument('--limit', type=int, default=60)
    ap.add_argument('--n_sp', type=int, nargs='+', default=[32, 64, 128])
    ap.add_argument('--methods', nargs='+', default=['fps', 'kmeans', 'spectral'])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--bs', type=int, default=15)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'

    for c in args.classes:
        d = load_class(c, 'ViT-B/16')
        n = min(args.limit, d['feat'].shape[0]) if args.limit else d['feat'].shape[0]
        P = seg_num[cat2id[c]]
        vw = torch.tensor(best_vweight[c], device=dev)
        print(f'\n===== {c}  ({n} shapes, {P} parts)')

        # ---- 1. partition ceiling ----
        print(f'{"oracle IoU":>14s}' + ''.join(f'{f"{m}/{k}":>16s}' for m in args.methods for k in args.n_sp))
        row, regs = [], []
        for m in args.methods:
            for k in args.n_sp:
                model = PartGeoZeV2(**{**best_param_v2, 'part': m, 'n_sp': k}).to(dev).eval()
                ious, R = [], []
                for s in range(0, n, args.bs):
                    xyz, nrm, fp = (d[q][s:s + args.bs].to(dev) for q in ('pc', 'normal', 'fpfh'))
                    lab = d['label'][s:s + args.bs].to(dev)
                    B = xyz.shape[0]
                    idx = model.graph(xyz)
                    seg = model.partition(xyz, F.normalize(nrm, dim=-1), hist_embed(fp), idx)
                    pred = oracle_pred(seg, lab.reshape(-1), P).view(B, PC_NUM)
                    for b in range(B):
                        pi = []
                        for part in range(P):
                            I = ((pred[b] == part) & (lab[b] == part)).sum().item()
                            U = ((pred[b] == part) | (lab[b] == part)).sum().item()
                            pi.append(1.0 if U == 0 else I / U)
                        ious.append(np.mean(pi))
                    R.append((int(seg.max()) + 1) / B)
                row.append(100 * np.mean(ious)); regs.append(np.mean(R))
        print(f'{"":>14s}' + ''.join(f'{v:>16.2f}' for v in row))
        print(f'{"regions":>14s}' + ''.join(f'{v:>16.1f}' for v in regs))

        # ---- 2. merge criterion on the default partition ----
        model = PartGeoZeV2(**best_param_v2).to(dev).eval()
        cos_raw, cos_cen, gate, gfpfh, same = [], [], [], [], []
        for s in range(0, n, args.bs):
            feat = vanilla_upprojection(d['feat'][s:s + args.bs].to(dev), d['ifseen'][s:s + args.bs].to(dev),
                                        d['pointloc'][s:s + args.bs].to(dev), img_size=(224, 224),
                                        n_points=PC_NUM, vweights=vw)[0].float()
            xyz, nrm, fp = (d[q][s:s + args.bs].to(dev) for q in ('pc', 'normal', 'fpfh'))
            lab = d['label'][s:s + args.bs].to(dev).reshape(-1)
            B = xyz.shape[0]
            nrm = F.normalize(nrm, dim=-1)
            idx = model.graph(xyz)
            i, j = flat_pairs(idx)
            seg = model.partition(xyz, nrm, hist_embed(fp), idx)
            S = int(seg.max()) + 1
            fn = F.normalize(feat.reshape(B * PC_NUM, -1), dim=-1)
            fc = model.merge_feats(fn, B, PC_NUM)
            z_raw = F.normalize(seg_mean(fn, seg, S), dim=-1)
            z_cen = F.normalize(seg_mean(fc, seg, S), dim=-1)
            votes = torch.bincount(seg * P + lab, minlength=S * P).view(S, P)
            maj = votes.argmax(1)
            gattn = model.gattn
            gattn.knn_sp = 0
            src, dst, b = gattn.region_graph(nrm.reshape(-1, 3), seg, S, i, j)
            keep = src < dst
            src, dst, b = src[keep], dst[keep], b[keep]
            g_reg = F.normalize(seg_mean(hist_embed(fp).reshape(B * PC_NUM, -1), seg, S), dim=-1)
            cos_raw.append((z_raw[src] * z_raw[dst]).sum(-1))
            cos_cen.append((z_cen[src] * z_cen[dst]).sum(-1))
            gate.append(b)
            gfpfh.append((g_reg[src] * g_reg[dst]).sum(-1))
            same.append(maj[src] == maj[dst])
        cos_raw, cos_cen, gate, gfpfh, same = (
            torch.cat(v) for v in (cos_raw, cos_cen, gate, gfpfh, same))
        print(f'adjacent pairs: {same.numel()}  same-part {same.float().mean():.2f}')
        print(f'{"score":>14s} {"same-part":>14s} {"diff-part":>14s} {"AUC":>7s}')
        for name, v in (('cos raw', cos_raw), ('cos centred', cos_cen), ('gate b_mn', gate),
                        ('FPFH region', gfpfh), ('gate x FPFH', gate * gfpfh)):
            print(f'{name:>14s} {stat(v, same):>14s} {stat(v, ~same):>14s} {auc(v, same):>7.3f}')
        print('admissible fraction (same / diff) at th_f, with th_n=0.3 on the gate:')
        for th in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            ok = (cos_cen >= th) & (gate >= 0.3)
            ok_r = (cos_raw >= th) & (gate >= 0.3)
            print(f'   th_f={th:.1f}   centred {ok[same].float().mean():.2f} / {ok[~same].float().mean():.2f}'
                  f'     raw {ok_r[same].float().mean():.2f} / {ok_r[~same].float().mean():.2f}')


if __name__ == '__main__':
    main()
