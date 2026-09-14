"""Evaluate PartGeoZe v2 and its baselines on the cached ShapeNetPart features.

Mirrors the loop of post_search.up_run_epoch (same unprojection, same text table, same per-shape
IoU) but takes the aggregation as a mode:

    point     per-point argmax of the unprojected multi-view CLIP feature (no aggregation)
    meanpool  mean pooling inside each superpoint            (the baseline v2 has to beat)
    v2        PartGeoZeV2: superpoints -> hierarchical merging -> recovery
    oracle    every superpoint takes its majority ground-truth part — the partition's ceiling

Timing covers the aggregation only (partition + merge + recovery), not the unprojection.
"""
import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F

from libs.lib_metric import calculate_shape_IoU, seg_num
from partseg.partclip import clip
from partseg.partmodel.best_param import best_vweight
from partseg.partmodel.partgeozev2 import PartGeoZeV2
from partseg.partmodel.post_search import (cat2id, index_start, simple_prompts, textual_encoder,
                                           view_tokens)
from partseg.partmodel.spectral import hist_embed
from partseg.rendering.unprojection import vanilla_upprojection
from semseg.semmodel.common import seg_mean

PC_NUM = 2048


def load_class(class_choice, model_name, layout='repo'):
    """The cached tensors of one category, on the CPU."""
    p = 'output/{}/{}'.format(model_name.replace('/', '_'), class_choice)
    ld = lambda n: torch.load(osp.join(p, f'test_{n}.pt'), map_location='cpu')  # noqa: E731
    return dict(feat=view_tokens(ld('features'), layout=layout), label=ld('labels') - index_start[cat2id[class_choice]],
                ifseen=ld('ifseen'), pointloc=ld('pointloc'), pc=ld('pc'), normal=ld('normal'),
                fpfh=ld('fpfh'))


def oracle_pred(reg, label, n_parts):
    """Majority ground-truth part per region.  reg, label: flat [B*N]."""
    R = int(reg.max()) + 1
    votes = torch.bincount(reg * n_parts + label, minlength=R * n_parts).view(R, n_parts)
    return votes.argmax(1)[reg]


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(class_choice, model_name, mode, params, device='cuda:0', bs=15, img_size=(224, 224),
             vweights=None, model=None, limit=0, layout='repo', prompt='repo'):
    d = load_class(class_choice, model_name, layout=layout)
    n = d['feat'].shape[0] if not limit else min(limit, d['feat'].shape[0])
    n_parts = seg_num[cat2id[class_choice]]

    clip_model, _ = clip.load(model_name, device=device)
    clip_model.eval()
    text, _ = textual_encoder(clip_model, class_choice,
                              simple_prompts(class_choice) if prompt == 'simple' else None,
                              device=device)
    text = F.normalize(text.float(), dim=-1)
    vw = torch.tensor(best_vweight[class_choice] if vweights is None else vweights, device=device)
    if model is None:
        model = PartGeoZeV2(**params).to(device).eval()

    preds, labels, ms, nreg = [], [], [], []
    for s in range(0, n, bs):
        e = min(s + bs, n)
        feat = d['feat'][s:e].to(device)
        feat = vanilla_upprojection(feat, d['ifseen'][s:e].to(device), d['pointloc'][s:e].to(device),
                                    img_size=img_size, n_points=PC_NUM, vweights=vw)[0].float()
        xyz, nrm, fpfh = (d[k][s:e].to(device) for k in ('pc', 'normal', 'fpfh'))
        label = d['label'][s:e].to(device)
        B = xyz.shape[0]

        sync(device); t0 = time.time()
        if mode == 'point':
            out, reg, z = F.normalize(feat, dim=-1), None, None
        else:
            idx = model.graph(xyz)
            seg = model.partition(xyz, F.normalize(nrm, dim=-1), hist_embed(fpfh), idx)
            if mode == 'v2':
                out, _, reg, z = model(xyz, nrm, fpfh, feat, seg=seg)
                reg = reg.reshape(-1)
            else:                                            # meanpool / oracle share the partition
                reg, out = seg, None
                fn = F.normalize(feat.reshape(B * PC_NUM, -1), dim=-1)
                w = (fn.norm(dim=-1) > 0).to(fn.dtype)
                z = F.normalize(seg_mean(fn, reg, int(reg.max()) + 1, w=w), dim=-1)

        # Classify at the SUPERPOINT level and propagate the label, whenever the per-point
        # feature is just its region's vector broadcast back (z is not None).  That is R x D
        # instead of N x D against the text table -- ~32x fewer multiply-adds at 64 regions per
        # 2048 points -- and the argmax of a broadcast vector is the broadcast of the argmax, so
        # it is exact, not an approximation.  (Measured: 0.005 vs 0.007 ms/shape.  Neither is
        # where the time goes; see bench_part.py.)
        if mode == 'oracle':
            pred = oracle_pred(reg, label.reshape(-1), n_parts).view(B, PC_NUM)
        elif z is not None:
            pred = (z @ text.t()).argmax(-1)[reg].view(B, PC_NUM)
        else:
            pred = (out @ text.t()).argmax(-1)
        sync(device); ms.append((time.time() - t0) * 1e3 / B)
        if reg is not None:
            nreg.append((int(reg.max()) + 1) / B)
        preds.append(pred.cpu()); labels.append(label.cpu())

    pred = torch.cat(preds).numpy(); lab = torch.cat(labels).numpy()
    acc = float((pred == lab).mean() * 100.0)
    cls = np.full(pred.shape[0], cat2id[class_choice])
    ious = np.array(calculate_shape_IoU(pred, lab, cls, class_choice, eva=True)[0])
    return dict(acc=acc, iou=float(ious.mean() * 100.0), n=int(pred.shape[0]), shape_ious=ious.tolist(),
                ms=float(np.mean(ms[1:] if len(ms) > 1 else ms)), regions=float(np.mean(nreg)) if nreg else 0.0)


def summarize(results):
    """Class-mean and instance-mean IoU over the categories in `results`."""
    cls_iou = float(np.mean([r['iou'] for r in results.values()]))
    all_ious = np.concatenate([r['shape_ious'] for r in results.values()])
    inst_iou = float(all_ious.mean() * 100.0)
    acc = float(np.mean([r['acc'] for r in results.values()]))
    ms = float(np.mean([r['ms'] for r in results.values()]))
    return dict(class_miou=cls_iou, instance_miou=inst_iou, acc=acc, ms=ms)
