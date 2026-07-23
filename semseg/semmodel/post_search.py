"""Text-side of the scene pipeline (mirrors partseg/partmodel/post_search.py).

partseg searches prompts per shape category; for the scene datasets the prompt table is fixed
(see best_param.py — every alternative we searched on ScanNet scored worse), so this module
builds and caches it per dataset.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from semseg.semmodel.best_param import LABELSET, best_prompt, text_encoder

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')


@torch.no_grad()
def textual_encoder(prompts, labelset, model_name):
    """[C, D] L2-normalised CLIP text table, ensembled over the `prompts` templates.

    The tower must match the one the 2D backbone was aligned to (ViT-B/32 for LSeg, ViT-L/14
    for OpenSeg), otherwise the cosines against the fused features are meaningless.
    """
    from transformers import CLIPModel, CLIPTokenizer
    clip = CLIPModel.from_pretrained(model_name).eval()
    tok = CLIPTokenizer.from_pretrained(model_name)
    per = []
    for tmpl in prompts:
        enc = tok([tmpl.format(c) for c in labelset], return_tensors='pt',
                  padding=True, truncation=True)
        f = clip.text_projection(clip.text_model(**enc, return_dict=True).pooler_output).float()
        per.append(F.normalize(f, dim=-1))
    return F.normalize(torch.stack(per).mean(0), dim=-1)


def search_prompt(dataset='scannet20', only_evaluate=True):
    """Return the text table for `dataset`, building and caching it on first use."""
    path = os.path.join(CACHE, f'text_{dataset}.npy')
    if os.path.exists(path) and only_evaluate:
        return torch.from_numpy(np.load(path))
    os.makedirs(CACHE, exist_ok=True)
    text = textual_encoder(best_prompt[dataset], LABELSET[dataset], text_encoder[dataset])
    np.save(path, text.numpy().astype(np.float32))
    return text
