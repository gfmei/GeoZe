# SemGeoZe v2 on nuScenes lidarseg

Outdoor LiDAR counterpart of the ScanNet pipeline ([semseg/README.md](README.md)). Same model,
same code path — three properties of the data force every configuration difference, and two setup
choices dominate the accuracy far more than the aggregation does.

**Headline: on nuScenes SemGeoZe v2 reaches parity with region mean pooling, not an improvement.**
It matches the baseline exactly at `th_f 0.98` and sits marginally below it at looser thresholds.
Reported as measured.

## Data

OpenScene-native layout, everything index-aligned to the aggregated LiDAR scan:

```
<nusc_root>/val/<scene>.pth        tuple (coords [N,3] f64, 0, labels [N] f64, 255 = ignore)
<nusc_root>/val/<scene>_spt.npy    VCCS supervoxel ids [N] i32 — 1024 per scan, precomputed
<feat_root>/<scene>.pt             {"feat": [V,768] f16, "mask_full": [N] bool}
```

Defaults in [`nuscenes.py`](nuscenes.py) point at:

```
NUSC_ROOT = /leonardo_work/IscrC_ERAR/data/nuscenes/nuscenes_3d
FEAT_ROOT = /leonardo_work/IscrC_ERAR/data/nuscenes/nuscenes_multiview_openseg_val
```

6018 val scans, ~278k points each (multi-sweep aggregated), 84 GB of geometry + 165 GB of fused
features. The released feature split is **val only**, so `scene_list` enumerates the scans that
actually have a `.pt`.

**No pre-pass is needed.** Unlike ScanNet there is no `sem_prep.py` step: normals are derived on
GPU from the same neighbour graph the merge already builds
([`common.estimate_normals`](semmodel/common.py)), and the supervoxels ship with the dataset.

## Running

```bash
python semseg/sem_run.py --dataset nuscenes                     # full val (6018 scans)
python semseg/sem_run.py --dataset nuscenes --stride 20         # 301-scan subset, ~3 min
python semseg/sem_run.py --dataset nuscenes --baseline meanpool # region mean pooling
python semseg/sem_run.py --dataset nuscenes --baseline point    # raw per-point features
python semseg/sem_run.py --dataset nuscenes --th_f 0.98         # the parity setting

python semseg/bench_speed.py --dataset nuscenes                 # timing
python semseg/probe_vocab.py                                    # label-set / prompt ablation
```

## Results — 301 val scans (stride 20)

| | mIoU | mAcc | OA |
|---|---|---|---|
| per-point argmax | 0.2882 | 0.4895 | 0.5079 |
| region mean pooling *(baseline)* | **0.2995** | 0.5150 | 0.6102 |
| SemGeoZe v2, `th_f` 0.85 | 0.2947 | 0.5045 | 0.6057 |
| SemGeoZe v2, `th_f` 0.90 | 0.2959 | 0.5028 | 0.6081 |
| SemGeoZe v2, `th_f` 0.95 | 0.2948 | 0.5036 | 0.6086 |
| SemGeoZe v2, `th_f` 0.98 | 0.2995 | 0.5136 | 0.6103 |

Region pooling is worth **+1.1 mIoU** over per-point classification (and +10 points of OA) —
that part transfers from ScanNet. Merging on top of it does not: every threshold lands at or below
the baseline and converges to a no-op as `th_f` → 1. Outdoor VCCS supervoxels are large and
semantically mixed, so enlarging the pooling support mixes classes instead of adding context.

Per class (`th_f` 0.85), showing the merge trading small gains against small losses rather than
finding anything systematic:

| class | mean pooling | SemGeoZe v2 | | class | mean pooling | SemGeoZe v2 |
|---|---|---|---|---|---|---|
| barrier | 0.067 | 0.071 | | trailer | 0.182 | 0.134 |
| bicycle | 0.143 | 0.143 | | truck | 0.382 | 0.394 |
| bus | 0.627 | 0.619 | | drivable surface | 0.604 | 0.594 |
| car | 0.463 | 0.453 | | other flat | 0.002 | 0.002 |
| construction vehicle | 0.151 | 0.167 | | sidewalk | 0.207 | 0.194 |
| motorcycle | 0.445 | 0.434 | | terrain | 0.272 | 0.260 |
| person | 0.149 | 0.154 | | manmade | 0.540 | 0.540 |
| traffic cone | 0.059 | 0.059 | | vegetation | 0.499 | 0.496 |

## Inference time (one A100, aggregation only)

| points/scan | mean pooling | KD-tree graph | multi-curve voting |
|---|---|---|---|
| 344k | 4.64 ms | 241.59 ms | **173.50 ms** |

## The two settings that actually matter

Both are already wired into [`best_param.py`](semmodel/best_param.py); they are recorded here
because each is worth several times more than the aggregation.

**1. The 43-label detail vocabulary is essential (+8.5 mIoU).** OpenScene does not classify against
the 16 eval names — it encodes a 43-entry detailed vocabulary and maps the argmax back:

| text table | mIoU | OA |
|---|---|---|
| 16 eval names, `"{}"` | 0.1926 | 0.3826 |
| 16 eval names, `"a {} in a scene"` | 0.1843 | 0.3536 |
| 43 detail names, `"{}"` | 0.2597 | 0.6068 |
| **43 detail names, `"a {} in a scene"`** | **0.2691** | **0.6089** |

*(31 scans, region mean pooling, ViT-L/14 tower.)* A VLM has never seen "manmade" or "other flat"
in a caption, but it has seen `building`, `pole`, `curb`, `grass`, `tree trunk`. Do not "simplify"
this back to the 16 names.

**2. The text tower must match the 2D backbone.** OpenSeg features are 768-d and live in CLIP
**ViT-L/14@336** space; ScanNet's LSeg features are 512-d and live in **ViT-B/32** space. Using
the wrong tower makes the cosines meaningless. `post_search.py` picks the tower per dataset.

## Why the configuration differs from ScanNet

| | ScanNet | nuScenes | consequence |
|---|---|---|---|
| colour | RGB per vertex | none | `w_c=0` — the colour cue is dropped, not fed zeros; the boundary gate falls back to normals |
| normals | open3d pre-pass | none shipped | derived on GPU from the kNN graph, no pre-pass at all |
| scale | ~5 m room | ~200 m scan | `curve_voxel` 0.02 → 0.20 m, `curve_max_dist` 0.3 → 3.0 m |
| feature coverage | 95% of points | **5.6%** of points | pooling fills the rest — this is why pooling beats per-point by so much here |
| labels | 95% of points | **7.3%** of points | multi-sweep aggregation labels only the keyframe; unlabelled points are ignored in the metric |
| partition | mesh segments (label-perfect) | VCCS supervoxels | no annotation leak, but a much less pure partition |

The feature/label sparsity is the most consequential row: the fused OpenSeg features cover only
5.6% of all points, but **75.9% of the labelled ones**.

## Known limitations

- **`other flat` is effectively dead (0.002 IoU)** under every configuration tried — the same kind
  of vocabulary collision as `shower curtain` on ScanNet.
- **`barrier` and `traffic cone` sit below 0.07.** Both are thin, small structures that a 5.6%
  feature coverage rarely lands on.
- Results above are a 301-scan subset. Full val is 6018 scans (~165 GB of feature reads); use
  `--stride 1` and expect roughly an hour of IO-bound runtime.
- The `@336` text tower is downloaded from HuggingFace on first use; set
  `HF_HOME` and pre-warm the cache for offline compute nodes.
