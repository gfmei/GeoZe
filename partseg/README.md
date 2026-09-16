# SimpleGeoZe — zero-shot 3D part segmentation

Object-level counterpart of [`semseg/`](../semseg/README.md), and the successor to the original
GeoZe part pipeline in [`partmodel/partgeoze.py`](partmodel/partgeoze.py). Same premise as GeoZe:
use the shape's geometry, with no trainable parameter, to refine a VLM's per-point features
before matching them to text.

```
partseg/
  simple_geoze.py       THE method, and the entry point
  part_run.py           caches the multi-view CLIP features; also runs the original GeoZe
  shapenet.py           ShapeNetPart, and the cached-feature loader
  partclip/ rendering/  the CLIP tower and the multi-view projection
  partmodel/
    partgeoze.py        the original GeoZe aggregation, unmodified
    best_param.py       searched prompts and view weights
    post_search.py      text table, cached-feature layout, the GeoZe evaluation loop
```

Point-graph primitives shared with `semseg/` — neighbour graph, gathers, segment mean, partition
cleanup — live in [`common/pointops.py`](../common/pointops.py), so there is one definition of
each rather than two that can drift.

## The method

```
1. geometry-only partition   k-means over position + normals + FPFH, then a few rounds of
                             affinity-weighted boundary refinement
2. pool per region           a masked mean of the unit VLM features
3. classify the REGIONS      S x D against the text table, not N x D
4. propagate the label       exact: the argmax of a broadcast vector is the broadcast of its argmax
```

Nothing touches the VLM feature until step 2, so a noisy feature can never corrupt the support it
is pooled over. Three details carry the partition:

- **FPFH as a cosine.** The histogram is L1-normalised and square-rooted, so an inner product
  between two of them is the Bhattacharyya coefficient.
- **Sign-free normals.** They enter through `n n^T`, whose entries are invariant to the arbitrary
  sign an estimator returns, so a flip cannot split a region.
- **Concavity.** Object parts meet at concave seams (seat/leg, wing/body), which is the
  local-convexity criterion, so a concave edge is penalised and a convex one is free. It needs
  consistently oriented normals, recovered by relaxing an Ising problem over the graph.

## Usage

```bash
# once per category: caches the CLIP pass over ten rendered views under output/
python part_run.py --classchoice all --datasetpath YOUR_SHAPENET_PATH --extract_only

python simple_geoze.py --classchoice all         # the method
python part_run.py --classchoice all             # the original GeoZe, for comparison
```

`--n_sp` sets the superpoint count, `--rounds` the boundary refinement, `--knn` the graph.

## Results — ShapeNetPart test, all 16 categories

Aggregation only (partition, pooling, classification), one A100, batches of 15 shapes.

| | class-mIoU | instance-mIoU | Acc | ms/shape |
|---|---|---|---|---|
| per-point argmax | 50.53 | 51.59 | 74.68 | 0.03 |
| farthest-point Voronoi + pooling | 53.59 | 54.91 | 76.99 | 1.07 |
| **SimpleGeoZe** | **54.82** | **55.59** | — | **1.62** |
| GeoZe (`partmodel/partgeoze.py`) | **56.12** | **57.18** | **78.37** | 40.30 |
| partition oracle | 85.91 | 86.49 | 95.17 | — |

**This is a speed result, not an accuracy one.** SimpleGeoZe is **1.30 class-mIoU behind GeoZe at
24x the speed**, and 4.3 ahead of classifying points directly.

Per category, the deficit is concentrated rather than spread: we win 5 of 16, and `cap` (−12.4 on
11 test shapes), `chair` (−5.0) and `laptop` (−4.9) account for almost the whole average. Nine
categories have fewer than 60 test shapes and four have fewer than 20, so class-mIoU here moves
several points on a handful of shapes.

## What this partition is not short of

The finding that shaped the design, and the reason the file is short:

| superpoints | oracle class-mIoU | end-task class-mIoU |
|---|---|---|
| 16 | 69.86 | 51.57 |
| 32 | 76.76 | **54.18** |
| 64 | 82.42 | 53.89 |
| 128 | 87.23 | 53.48 |

**A 17-point rise in what the partition admits buys at most 2 points of what the method
achieves**, and past the optimum it buys less than nothing: smaller regions average fewer
features, and that variance costs more than the raised ceiling gains. Roughly 31 class-mIoU
separate the result from its own oracle, and that gap is VLM/text misalignment, which no
aggregation can close. **Tune the superpoint count on the end task, never on oracle IoU.**

The sharpest form of this: a cut-pursuit partition is strictly purer (88.2 oracle at 126 regions
against 87.2 for k-means) and classifies **worse** (53.75 against 54.82), because it spends
regions where the geometry varies — 6.2% of them end up holding fewer than 8 points, against 2.0%
for k-means, and a region that pools nine features has a correspondingly noisy mean.

## What was measured and rejected

Each of these was implemented, run on the full test split, and removed. The numbers are here so
they are not re-tried; the code is in the git history (`git log -- partseg/`).

| | result |
|---|---|
| hierarchical merging (the `semseg` stage) | **−1.81** class-mIoU; no cue separates adjacent same-part from different-part regions (best AUC 0.67, worst 0.43, adjacent cosine 0.99 either way) |
| intra- and inter-region attention | within noise, as in `semseg` |
| cut pursuit, GPU rewrite of Landrieu & Obozinski | best oracle IoU of any partition, **−1.07** class-mIoU |
| sparse spectral (normalised cut) | −0.07 at 3x the time; strictly dominates the dense solve, which it replaces |
| VCCS, voxel graph (what `semseg` uses) | −0.9, at 23x the time; best boundary recall of any partition |
| VCCS on the kNN graph | −0.35, 8x faster than the voxel version |
| Nyström, LOBPCG | ties k-means at 5x the cost; slower than an exact decomposition |
| multi-curve kNN | −0.9 oracle; exact kNN is already 0.19 ms at 2048 points |
| Z-order k-means seeding | −0.41; the loop it removes is only 32 iterations at this K |

## A caveat for anyone reproducing the baseline

`part_run.Extractor` stores the cached feature maps channel-first and `post_search` reads them
with a reshape that interleaves channels with patch positions. The permuted read is provably the
intended one — its rows carry unit norm, as written, and adjacent patches correlate at 0.70 rather
than 0.00 — yet substituting it **lowers** per-point class-mIoU from 50.53 to 13.96, because the
released prompts and view weights were searched against the original behaviour and have absorbed
it. Under neutral prompts the ordering reverses (25.14 against 20.44). Everything here is
therefore evaluated under the released layout, which is the published footing; correcting it would
require re-running the prompt search. `view_tokens(feat, layout=...)` exposes both.
