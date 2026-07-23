# SemGeoZe v2 — zero-shot 3D scene segmentation

Scene-level counterpart of `partseg/`, same premise as GeoZe: use the point cloud's geometry,
with no trainable parameter, to refine a VLM's per-point features before matching them to text.
Supports **ScanNet v2** (LSeg features, 20 classes) and **nuScenes lidarseg** (OpenSeg features,
16 classes).

The headline is **speed**: 81.89 ms/scene against the 2125.61 ms GeoZe reports in paper Table 6
(~26x), because the aggregation is scatter-reduces over one edge list rather than Sinkhorn
attention inside a knn patch around every superpoint. See the root README for the timing table.

```
semseg/
  sem_run.py            entry point            (mirrors partseg/part_run.py)
  sem_prep.py           one-off geometry cache (ScanNet: normals, FPFH, VCCS)
  bench_speed.py        inference-time benchmark (paper Table 6 comparable)
  scannet.py            ScanNet dataset        (mirrors partseg/shapenet.py)
  nuscenes.py           nuScenes dataset       (see README_nuscenes.md)
  metrics.py            mIoU / mAcc / OA
  probe_vocab.py        label-set / prompt ablation
  export_demo.py        scene export for the docs/ viewer
  semmodel/
    semgeozev2.py         the model              (mirrors partseg/partmodel/partgeoze.py)
    best_param.py       class names, prompts, searched hyper-parameters
    post_search.py      CLIP text table
    common.py           segment-reduction helpers
    vccs.py             Voxel Cloud Connectivity Segmentation
    curve.py            multi-curve voting kNN graph
    serialization/      space-filling curve codes (z / Hilbert)
```

## Usage

```bash
# once: per-scene normals + FPFH + VCCS  (CPU, shardable, ~30 min over 12 shards)
python semseg/sem_prep.py --shard 0 --nshards 12

python semseg/sem_run.py                       # SemGeoZe v2, full 312-scene val
python semseg/sem_run.py --baseline meanpool   # region mean pooling, the baseline
python semseg/sem_run.py --baseline point      # raw per-point features
python semseg/sem_run.py --part vccs           # self-contained partition (no mesh segments)
python semseg/sem_run.py --curve               # multi-curve voting graph instead of a KD-tree

# nuScenes lidarseg (6018 val scans) — no pre-pass needed, normals come from the kNN graph
python semseg/sem_run.py --dataset nuscenes
python semseg/sem_run.py --dataset nuscenes --baseline meanpool

python semseg/bench_speed.py --dataset scannet   # reproduce the inference-time table
```

## Inference time (one A100, aggregation only)

| dataset | points/scene | mean pooling | KD-tree graph | multi-curve voting |
|---|---|---|---|---|
| ScanNet | 166k | 1.60 ms | 113.75 ms | **81.89 ms** |
| nuScenes | 344k | 4.64 ms | 241.59 ms | **173.50 ms** |

Voting nearly halves the graph cost versus an exact KD-tree (ScanNet 70.43 -> 38.98 ms) at equal
accuracy, and keeps everything on the GPU.

## Results — ScanNet v2 val, 312 scenes, LSeg fused features

| | partition | mIoU | mAcc | OA | regions | runtime |
|---|---|---|---|---|---|---|
| per-point argmax | — | 0.4972 | 0.6346 | 0.7598 | — | |
| region mean pooling *(baseline)* | mesh | 0.5564 | 0.6834 | 0.8147 | 1068 | |
| **SemGeoZe v2** | mesh | **0.5606** | **0.6856** | 0.8143 | 251 | 107 ms/scene |
| SemGeoZe v2 `--curve` | mesh | 0.5585 | 0.6834 | 0.8137 | 265 | 71 ms/scene |
| region mean pooling | vccs | 0.5229 | 0.6565 | 0.7916 | 936 | |
| SemGeoZe v2 | vccs | 0.5236 | 0.6555 | 0.7941 | 196 | |

## Results — nuScenes lidarseg val, 301 scans

| | mIoU | mAcc | OA |
|---|---|---|---|
| per-point argmax | 0.2882 | 0.4895 | 0.5079 |
| region mean pooling *(baseline)* | **0.2995** | 0.5150 | 0.6102 |
| SemGeoZe v2 (`th_f` 0.85) | 0.2947 | 0.5045 | 0.6057 |
| SemGeoZe v2 (`th_f` 0.98) | 0.2995 | 0.5136 | 0.6103 |

**On nuScenes the merge is neutral** — parity at `th_f` 0.98, marginally below at looser
thresholds, converging to a no-op as `th_f` -> 1. The defensible claim is "same accuracy as plain
region pooling, 173 ms/scan", not an improvement. Region *pooling* itself is still worth +1.1 mIoU
over per-point classification.

Two setup choices dominate nuScenes accuracy far more than the aggregation: the **43-label detail
vocabulary** (+8.5 mIoU over the 16 eval names) and the **matching text tower** (ViT-L/14@336 for
OpenSeg, not ViT-B/32). Full details, data layout, per-class results and the outdoor
configuration: **[README_nuscenes.md](README_nuscenes.md)**.

## What each stage is worth (ScanNet full val, vs. mean pooling at 0.5564)

| stage | Δ mIoU | why |
|---|---|---|
| `IntraConsensusAttn` | +0.0002 | regions are already 98.5% homogeneous in VLM space (`cos(f_i, z_mean)` = 0.985), so reweighting their points moves the region vector by `cos = 0.9993` and flips only 2.4% of region labels |
| `HierMerge` | **+0.0043** | the only stage with a consistent effect — it *enlarges* the pooling support instead of reweighting or mixing it |
| `InterStructuralAttn` | +0.0015 | within noise: its normal-based boundary gate reads 0.908, i.e. wide open, because adjacent mesh segments usually do have agreeing normals whether or not they are one object |

Intra and inter are therefore **off by default** (`--intra`, `--gamma0` to enable). The guiding
principle that survived measurement is narrower than "attention helps":

> Aggregation may enlarge a region only when the enlargement is structurally justified, and it
> must never average across regions that stay separate.

## Caveats to carry into any write-up

- The ScanNet **mesh over-segmentation is label-perfect** (assigning each segment its dominant GT
  label scores mIoU 1.0000) because annotators labelled these very segments. Part of the
  per-point → per-region jump is an annotation artifact. `--part vccs` has no such leak.
- Pooling over the *ground-truth* region reaches only 0.6845, so ~21 mIoU of the gap to the oracle
  is irreducible LSeg/text misalignment; only ~13 is reachable by any aggregation.
- `shower curtain` scores 0.000 and cannot be fixed from the text side — 87% of its points are
  predicted `curtain`; prompt ensembles and disambiguated names both made things worse.
- **+0.43 mIoU is small.** Subsets of ≤52 scenes swing mIoU by ±1, and several designs that looked
  like +1.5 there landed at ~0 on full val. Always evaluate on all 312 (~4 min on one A100).

## Where these numbers come from

The per-stage and per-graph ablations behind the tables above are kept as raw results in
`semseg/out/`:

| file | contents |
|---|---|
| `meanpool.json`, `semgeozev2*.json` | the headline runs, reproducible with the commands above |
| `ablation_stages_mesh.json` | point / meanpool / intra / intra+inter / hier / hier+intra, mesh segments |
| `ablation_stages_vccs.json` | the same ladder on the VCCS partition |
| `ablation_graph_vote_vs_kdtree.json` | KD-tree vs multi-curve voting at `min_votes` 4/5/6 |

Each entry stores mIoU, mAcc, OA and the 20 per-class IoUs, so the claims above can be checked
without re-running anything.
