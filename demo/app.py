"""SemGeoZe v2 — interactive HuggingFace Space.

Runs the real aggregation on CPU: the expensive part (fusing 2D VLM features onto the points)
is precomputed and shipped as a dataset, so the Space only does the training-free geometric
step — which is exactly the contribution, and cheap enough to run per request.

Move `th_f` and watch the merge: at 1.0 nothing merges and the output equals region mean
pooling; lowering it grows the pooling support. The mIoU readout is honest in both directions.

Layout expected in ./data (or the HF dataset given by GEOZE_DATA):
    <scene>.npz   xyz f32[N,3], rgb f32[N,3], normal f16[N,3], fpfh f16[N,33],
                  feat f16[V,512], fmask bool[N], label i64[N], seg i64[N]
    text.npy      [20,512] L2-normalised CLIP ViT-B/32 text table
Build it with demo/pack_space_data.py.
"""
import glob
import os
import time

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

from semgeozev2 import SemGeoZeV2, ConfusionMatrix, kdtree_graph, seg_mean, SCANNET_COLORS

DATA = os.environ.get("GEOZE_DATA", os.path.join(os.path.dirname(__file__), "data"))
SCENES = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DATA, "*.npz")))
TEXT = torch.from_numpy(np.load(os.path.join(DATA, "text.npy")))
MAX_DRAW = 60000                                   # points sent to the browser


def _load(scene):
    z = np.load(os.path.join(DATA, f"{scene}.npz"))
    return {k: z[k] for k in z.files}


def _fig(xyz, colors, title):
    fig = go.Figure(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        marker=dict(size=1.4, color=colors), hoverinfo="skip"))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        scene=dict(aspectmode="data", xaxis=dict(visible=False),
                   yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, t=30, b=0), height=520, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _hex(lab):
    out = np.full((len(lab), 3), 60, np.uint8)
    ok = lab < 20
    out[ok] = np.asarray(SCANNET_COLORS, np.uint8)[lab[ok]]
    return ["#%02x%02x%02x" % tuple(c) for c in out]


def run(scene, th_f, th_n, rounds, view):
    d = _load(scene)
    N = d["xyz"].shape[0]
    feat = np.zeros((N, d["feat"].shape[1]), np.float32)
    feat[d["fmask"]] = d["feat"].astype(np.float32)
    feat = F.normalize(torch.from_numpy(feat), dim=-1)
    valid = torch.from_numpy(d["fmask"])
    seg = torch.from_numpy(d["seg"])
    S = int(d["seg"].max()) + 1
    xyz = torch.from_numpy(d["xyz"])
    rgb = torch.from_numpy(d["rgb"])
    nrm = torch.from_numpy(d["normal"].astype(np.float32))
    fpfh = torch.from_numpy(d["fpfh"].astype(np.float32))

    model = SemGeoZeV2(th_f=float(th_f), th_n=float(th_n), rounds=int(rounds)).eval()
    t0 = time.time()
    pairs = kdtree_graph(d["xyz"], 8, "cpu")
    with torch.no_grad():
        out, _, reg = model(xyz, rgb, nrm, fpfh, feat, valid, seg, pairs)
    dt = time.time() - t0

    gt = d["label"]
    z = F.normalize(seg_mean(feat, seg, S, w=valid.float()), dim=-1)
    preds = {
        "SemGeoZe v2": (out @ TEXT.T).argmax(-1).numpy(),
        "region mean pooling": (z @ TEXT.T).argmax(-1).numpy()[d["seg"]],
        "per-point VLM": (feat @ TEXT.T).argmax(-1).numpy(),
    }
    miou = {}
    for k, p in preds.items():
        cm = ConfusionMatrix(20); cm.add(gt, p); miou[k] = cm.scores()["miou"]

    R = int(reg.max().item()) + 1
    step = max(1, N // MAX_DRAW)
    sub = slice(None, None, step)
    if view == "merged regions":
        h = (reg.numpy()[sub].astype(np.uint64) * 2654435761) % (1 << 24)
        col = ["#%06x" % int(v) for v in h]
        title = f"{S} segments merged to {R} regions"
    elif view == "ground truth":
        col = _hex(np.where((gt >= 0) & (gt < 20), gt, 255)[sub]); title = "ground truth"
    elif view == "region mean pooling":
        col = _hex(preds["region mean pooling"][sub]); title = "region mean pooling"
    else:
        col = _hex(preds["SemGeoZe v2"][sub]); title = "SemGeoZe v2"

    md = (
        f"**{scene}** — {N:,} points, **{S} &rarr; {R}** regions, "
        f"aggregation **{dt*1000:.0f} ms** on this Space's CPU\n\n"
        f"| method | scene mIoU |\n|---|---|\n"
        + "\n".join(f"| {k} | {v:.4f} |" for k, v in miou.items())
        + f"\n\n*{'Merging is active' if R < S else 'No merge at this threshold — output equals region mean pooling'}.*"
    )
    return _fig(d["xyz"][sub], col, title), md


DESC = """
# SemGeoZe v2 — training-free 3D scene segmentation

Refines a vision–language model's per-point features using **only the point cloud's geometry** —
no training, no learned parameters. Adjacent regions merge in mutual-best-match pairs when they
agree both semantically and geometrically, which enlarges the pooling support without ever
averaging two regions the geometry keeps apart.

The 2D feature fusion is precomputed; what runs here is the aggregation itself — 82 ms/scene on
an A100, a couple of seconds on this Space's CPU, against 2125.61 ms for GeoZe's scene pipeline
(paper Table 6).

**Try:** set `th_f` to 1.0 (nothing merges — output is exactly region mean pooling), then lower it
and watch regions coalesce. On full ScanNet val, SemGeoZe v2 scores 0.5606 mIoU vs 0.5564 for region
mean pooling; per-scene it moves in both directions.
"""

with gr.Blocks(title="SemGeoZe v2", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESC)
    with gr.Row():
        with gr.Column(scale=1):
            scene = gr.Dropdown(SCENES, value=SCENES[0] if SCENES else None, label="scene")
            view = gr.Radio(["SemGeoZe v2", "region mean pooling", "ground truth", "merged regions"],
                            value="SemGeoZe v2", label="view")
            th_f = gr.Slider(0.60, 1.00, 0.85, step=0.01, label="th_f · semantic merge threshold")
            th_n = gr.Slider(0.00, 0.90, 0.30, step=0.05, label="th_n · geometric merge threshold")
            rounds = gr.Slider(0, 12, 10, step=1, label="rounds · hierarchy depth")
            btn = gr.Button("Run SemGeoZe v2", variant="primary")
            info = gr.Markdown()
        with gr.Column(scale=2):
            plot = gr.Plot()
    inputs = [scene, th_f, th_n, rounds, view]
    btn.click(run, inputs, [plot, info])
    for c in (scene, view):
        c.change(run, inputs, [plot, info])
    demo.load(run, inputs, [plot, info])

if __name__ == "__main__":
    demo.launch()
