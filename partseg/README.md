# PartGeoZe v2 — zero-shot 3D part segmentation

Object-level counterpart of [`semseg/`](../semseg/README.md), and the successor to the original
GeoZe part pipeline in [`partmodel/partgeoze.py`](partmodel/partgeoze.py). Same premise as GeoZe:
use the shape's geometry, with no trainable parameter anywhere, to clean up a VLM's per-point
features before they are matched against text.

```
partseg/
  part_run.py           entry point: feature extraction + every evaluation mode
  run_final.sh          the comparison table, one row per configuration
  run_layout.sh         the cached-feature-layout ablation
  report.py             collects out/*.json into the tables below
  bench_part.py         stage-by-stage timing of the partition
  probe_superpoints.py  partition QUALITY sweep (oracle IoU, undersegmentation, BR/BP)
  probe_partition.py    can the merge criterion tell parts apart?  (it cannot; see below)
  shapenet.py           ShapeNetPart dataset
  partmodel/
    partgeozev2.py      the model
    spectral.py         superpoints: affinity, Nystrom embedding, k-means, cleanup
    eval_v2.py          batched evaluation shared by every mode
    best_param.py       prompts, view weights, and the searched v2 hyper-parameters
    post_search.py      text table, feature-map layout, the original GeoZe evaluation
    partgeoze.py        the original GeoZe aggregation, unmodified
```

Space-filling curve codes come from [`libs/serialization/`](../libs/serialization) — the same
shared copy `semseg/semmodel/curve.py` uses.

## The pipeline

```
spectral superpoints          XYZ + normals + FPFH, never the VLM feature
  -> region mean pooling      z_m = Norm(mean of the region's unit features)
  -> [optional] HierMerge     adjacency-constrained mutual-best-match agglomeration
  -> [optional] InterAttn     boundary-gated residual between regions
  -> classify at region level and propagate the label to points
```

**The partition never sees the VLM feature**, so a noisy feature can not corrupt the regions it
will later be pooled over. This is the same rule `semseg/sem_prep.py` follows for scenes.

### Superpoints

`--part` selects the method. All four are built from geometry alone and never from the VLM
feature, so a noisy feature can not corrupt the regions it will later be pooled over.

| `--part` | what it is | where it runs |
|---|---|---|
| `kmeans` *(default)* | weighted k-means in the cue space below | GPU, batched |
| `spectral` | normalised cut; `--embed sparse\|dense\|nystrom\|lobpcg` | GPU, batched |
| `vccs_gpu` | the VCCS algorithm on the point kNN graph | GPU, batched |
| `vccs` | Voxel Cloud Connectivity Segmentation — what `semseg` uses on ScanNet | CPU, per shape |
| `fps` | farthest-point Voronoi cells (the weakest baseline) | GPU, batched |

A shape is ~2k points, so the affinity is dense and batched. On the symmetric kNN graph,

    w_ij = exp( - sum_cue  w_cue * d_cue(i,j) / tau_cue )

with four cues, each divided by its own mean over the shape's edges so there is no temperature
to tune:

| cue | distance | note |
|---|---|---|
| position | `\|\|x_i - x_j\|\|^2` | optionally Zelnik-Manor local scaling, `/(sigma_i sigma_j)` |
| normal | `1 - \|n_i . n_j\|` | sign-free: estimated normals are unoriented |
| FPFH | `1 - <g_i, g_j>` | `hist_embed` L1-normalises then square-roots the histogram, so the inner product is the Bhattacharyya coefficient |
| **concavity** | `relu(-c_ij)` | `c_ij = (m_i - m_j).(x_i - x_j)/\|\|dx\|\|` with CONSISTENTLY ORIENTED normals `m` |

The concavity cue is the one with a principled claim on part boundaries: **object parts meet at
concave seams** (seat/leg, wing/body), which is the LCCP criterion. Only concavity is penalised;
a convex edge costs nothing. It needs oriented normals, and the cached ones are unoriented, so
`orient_normals` resolves the signs by relaxing the Ising problem `max sum_ij s_i s_j (n_i.n_j)`
over the kNN graph, seeded outward from the shape centroid. It converges by iteration five and
reaches 0.94 edge agreement on airplane, 0.88 on chair, 0.86 on motorbike, 0.84 on table — thin
open surfaces (table legs, lamp wires) are where orientation stays genuinely ambiguous, and
`probe_superpoints.py` prints the agreement per class rather than assuming it.

The embedding is the top eigenvectors of `D^-1/2 W D^-1/2`, rows L2-normalised, then k-means.
Afterwards every cluster is split into its connected components and fragments below `min_size`
are absorbed by a neighbour, so a region is always spatially connected.

### Why it is not the obvious O(N^3)

`bench_part.py` prints where a shape's time goes. The answer is: **almost entirely in the
eigendecomposition**, and nowhere near the features.

    pooling                     0.024 ms/shape
    classify at region level    0.005 ms/shape   (per-point would be 0.007)
    dense eigh, 2048 x 2048    35.1   ms/shape   <- 93% of the total

So the aggregation was never the cost, and classifying 64 region vectors instead of 2048 point
vectors — which is exact, since the argmax of a broadcast vector is the broadcast of the argmax —
saves 0.002 ms. What actually pays:

| change | ms/shape | what it does |
|---|---|---|
| dense | 37.56 | one N x N eigendecomposition per shape |
| + Nystrom, 256 landmarks | 17.40 | decompose an m x m block, extend to every point |
| + Hilbert-curve landmarks | 10.69 | `farthest_point_sample` is a Python loop over the sample count (2.89 -> 0.27 ms) |
| + plain extension | 5.40 | drops the second m x m eigendecomposition |
| + 128 landmarks | 2.88 | |

Nystrom keeps a per-point embedding row, so region boundaries stay at full resolution — unlike a
cluster-the-coarse-set-then-propagate scheme. Two details worth knowing: the row-sum estimate
solves for one vector instead of forming a pseudo-inverse, and `torch.linalg.eigh` on CUDA returns
eigenvectors **column-major**, a stride that survives slicing and then breaks the Hilbert encoder's
dtype-view.

`--embed sparse` is the spectral path that is actually worth running: the operator is only ever
**applied** — a gather for `W x` and the matching scatter-add for `W^T x` — so nothing N x N is
ever allocated, and subspace iteration with Cholesky-QR extracts the leading eigenvectors
directly. At matched region counts it equals or beats the full decomposition at 2-6x less time,
which makes `--embed dense` strictly dominated. Two approximations that did **not** pay off are
kept only so the measurement is reproducible: Nystrom needs 256 landmarks and 10.6 ms merely to
tie plain k-means, and `torch.lobpcg` on the batched dense matrix runs at 189 ms/shape, slower
than the decomposition it replaces.

The **multi-curve kNN** (`curve_knn`, the object-level counterpart of `semseg/semmodel/curve.py`)
is likewise available and likewise not the default: at 2048 points the exact distance matrix costs
0.19 ms, so there is no time to win, and 0.89 recall against the true neighbours costs 0.9 oracle
IoU and ~10 points of boundary recall. It is the right structure at scene scale, not at object
scale.

### VCCS at object scale

`--part vccs` runs the algorithm `semseg/sem_prep.py` uses on ScanNet rooms, through the vendored
[`semseg/semmodel/vccs.py`](../semseg/semmodel/vccs.py). Three things differ at object scale and
`vccs_superpoints` handles all three:

* **No colour.** ShapeNetPart ships geometry only, so the colour term of Eq. 1 is switched off
  rather than fed zeros, which would make every pair look identical on that cue.
* **Scale.** The released room settings assume metres. A ShapeNet shape is unit-sphere normalised
  with ~0.05 point spacing at 2048 points, and `vccs_voxel` must be at least that or the
  26-connectivity adjacency falls apart and the BFS cannot grow. Do not copy the `semseg` value.
* **A target count.** `vccs_seed='fps'` asks for a specific supervoxel count, so VCCS is
  comparable at matched region counts rather than at some seed resolution.

It works, and on this task it is **dominated** — worth stating plainly, because VCCS is the
obvious thing to reach for given `semseg`:

| partition | ~regions | oracle IoU | boundary recall | end-task class-mIoU | ms/shape |
|---|---|---|---|---|---|
| VCCS | 78.1 | 81.37 | **86.50** | 53.92 | 36.71 |
| k-means + refinement | 66.4 | 83.00 | 72.73 | **54.18** | **2.57** |
| sparse spectral | 79.0 | **85.03** | 75.62 | **54.44** | 8.54 |

VCCS wins **boundary recall** by a wide margin — compact BFS-grown supervoxels hug geometric
edges — but loses ~3.7 oracle IoU at a matched region count and ~0.5 class-mIoU on the end task,
at 4–15x the cost, because it is CPU numpy while the others are batched on the GPU. Its
boundary-aware BFS (`vccs_boundary`, the analogue of the concavity cue) did not help either:
81.03 against 81.37.

### The same algorithm, batched on the GPU

Almost all of the CPU version's cost is structural, not algorithmic: voxelise, build a CSR
adjacency over the occupied voxels, then walk a python-level BFS. But nothing in VCCS needs the
*voxel* grid specifically — it needs a connectivity graph, a distance to the seed, and a rule for
resolving concurrent claims. We already build a kNN graph on the GPU, so `vccs_gpu` ports all
three directly: Eq. 1 without the colour term, flow-constrained growth so regions stay connected,
claims resolved by minimum distance, and each seed moving to the member nearest its centroid.
Every step is a gather or a scatter over `[B,N,k]`.

| | ~regions | oracle IoU | boundary recall | end-task class-mIoU | ms/shape |
|---|---|---|---|---|---|
| `vccs` (CPU, voxel graph) | 78.1 | 81.37 | **86.50** | 53.92 | 36.71 |
| `vccs_gpu` (kNN graph) | 63.7 | **83.29** | 69.99 | **54.47** | **5.09** |

**8x faster, better oracle IoU, and the best end-task number of any partition here** — but it
gives up precisely what made VCCS distinctive. Boundary recall falls 86.50 → 69.99, because the
voxel 26-neighbourhood is a *lattice* adjacency that grows compact blobs with many short edges,
while a kNN graph is a *surface* adjacency that grows smooth regions following the shape. The port
is therefore not a drop-in replacement: it is faster and scores better, and it is a different
partition. Both are kept, and `--part vccs` remains the reference when boundary adherence is what
you want.

`vccs_gpu` is the **best end-task partition measured here** (54.47), but only by 0.03 over sparse
spectral and 0.29 over the k-means default — which is itself the point: see
[the partition is not what limits the end task](#the-partition-is-not-what-limits-the-end-task).

## Usage

```bash
# feature extraction is cached under output/<model>/<category>/ and skipped when present
python part_run.py --classchoice all --baseline point       # per-point argmax, no aggregation
python part_run.py --classchoice all --baseline meanpool    # pooling over the v2 superpoints
python part_run.py --classchoice all --baseline oracle      # the partition's ceiling
python part_run.py --classchoice all --model v2             # PartGeoZe v2
python part_run.py --classchoice all --model geoze          # the original GeoZe aggregation

bash run_final.sh                      # every row of the comparison, into out/
python report.py                       # collect out/*.json into the tables
python bench_part.py --cls chair       # stage-by-stage timing
python probe_superpoints.py            # partition quality sweep
python probe_partition.py              # merge-criterion analysis
```

Partition knobs: `--part kmeans|spectral|vccs|fps`, `--embed sparse|dense|nystrom|lobpcg`,
`--n_sp`, `--w_v` (concavity), `--refine`, `--knn`, `--no_self_tune`, `--n_land`, `--ortho`,
`--sparse_iters`, `--vccs_voxel`, `--vccs_seed`, `--vccs_boundary`. Merge knobs: `--rounds`,
`--th_f`, `--th_n`. Everything defaults from `partmodel/best_param.py` — note `refine=3` is
already on there, so `--refine 0` is what turns boundary refinement off.

## Two findings that constrain any claim made here

### The merge stage does not transfer from scenes

SemGeoZe v2's load-bearing stage is adjacency-constrained mutual-best-match merging. On parts it
**costs** accuracy. `probe_partition.py` scores every adjacent superpoint pair by whether the two
regions carry the same ground-truth part:

| cue | best AUC over ten categories | worst |
|---|---|---|
| raw feature cosine | 0.615 | 0.430 |
| per-shape mean-centred cosine | 0.627 | 0.460 |
| normal boundary gate | 0.669 | 0.493 |
| region FPFH cosine | 0.540 | 0.540 |

Adjacent regions sit at cosine 0.99 whether or not they share a part, and on `table` the semantic
cue is *below* chance. Nothing to gate on, so merging mixes parts. `rounds=0` is the default; the
implementation is kept and switchable with `--rounds 10`.

### The cached feature-map layout is buggy, and the released prompts are tuned to the bug

`part_run.Extractor` stores the multi-view CLIP maps channel-first, `[n, 10, 512, 14, 14]`, and
`post_search` read them with `reshape(-1, 10, 196, 512)`, which interleaves channels with patch
positions. The permuted read is provably correct — its rows have unit norm, which is what the
extractor wrote, and adjacent-patch cosine 0.70 rather than 0.00. But:

| features | prompts | class-mIoU |
|---|---|---|
| repo layout | searched (`best_prompt`) | **50.53** |
| repo layout | `a {part} of a {category}` | 20.44 |
| true CLIP tokens | searched | 13.96 |
| true CLIP tokens | `a {part} of a {category}` | 25.14 |

With neutral prompts the correct layout wins by +4.7, so it *is* a bug — but `best_prompt` and
`best_vweight` were searched against the buggy pipeline and absorb it, so switching the layout
alone costs ~37 mIoU. Using it requires re-running the prompt search. `view_tokens(feat,
layout=...)` exposes both and defaults to `repo`, so every method here is compared on the
published footing. Reproduce with `bash run_layout.sh`.

## Results — ShapeNetPart test, all 16 categories

`class-mIoU` averages the 16 per-category IoUs; `instance-mIoU` averages over shapes. Timing is
the aggregation only (partition, pooling, classification), one A100, batches of 15 shapes.

| | class-mIoU | instance-mIoU | Acc | ms/shape |
|---|---|---|---|---|
| per-point argmax | 50.53 | 51.59 | 74.68 | 0.03 |
| farthest-point Voronoi + pooling, 64 | 53.59 | 54.91 | 76.99 | 1.07 |
| **PartGeoZe v2**, k-means + refinement, 32 | 54.18 | 55.26 | 77.32 | **2.57** |
| **PartGeoZe v2**, sparse spectral, 32 | **54.44** | **55.32** | **77.52** | 8.54 |
| VCCS + pooling, 64 | 53.92 | 55.08 | 77.21 | 36.71 |
| **VCCS on the kNN graph (`vccs_gpu`), 32** | **54.47** | **55.52** | 77.39 | 5.09 |
| GeoZe (`partgeoze.py`) | **56.12** | **57.18** | **78.37** | 40.30 |
| partition oracle (sparse, 48) | 85.79 | 87.04 | 95.43 | — |

**State the trade honestly: v2 does not beat GeoZe on accuracy.** It is 1.68 class-mIoU behind at
**4.7x** the speed, or 1.94 behind at **15.7x**. What it does beat is pooling over a naive
partition (+0.85) and per-point classification (+3.9). The contribution is speed and a better
partition, not accuracy.

### Partition quality, mean over 7 categories, at matched region counts

| ~regions | dense spectral | sparse spectral | k-means + refinement |
|---|---|---|---|
| 66-68 | 84.20 @ 37.3 ms | 83.69 @ 8.8 ms | 83.00 @ 2.2 ms |
| 98-103 | 85.93 @ 37.2 ms | 86.56 @ 19.5 ms | 85.95 @ 2.2 ms |
| 128-132 | 87.24 @ 37.2 ms | 87.78 @ 23.8 ms | 87.17 @ 2.1 ms |

Compare at matched **regions**, never at matched `n_sp`: a less converged embedding fragments
clusters, so sparse with 30 iterations returns 88.9 regions when asked for 64 and its oracle IoU
is inflated accordingly.

### The partition is not what limits the end task

Sweeping resolution moves the ceiling a long way and the result barely at all:

| superpoints | oracle class-mIoU | end-task class-mIoU |
|---|---|---|
| 16 | 69.86 | 51.57 |
| 32 | 76.76 | **54.18** |
| 64 | 82.42 | 53.89 |
| 128 | 87.23 | 53.48 |

**+17.4 oracle buys nothing**, and past 32 superpoints the end task gets *worse*: smaller regions
average fewer features, and that noise costs more than the higher ceiling gains. Roughly 31 points
separate the best result from its own oracle, and that gap is VLM/text misalignment, which no
partition can close. Tune `n_sp` on the end task, never on oracle IoU.

## What did not work

| idea | result | why |
|---|---|---|
| hierarchical merging (the SemGeoZe v2 stage) | **-1.81** class-mIoU | no cue separates adjacent same-part from different-part regions; best AUC 0.67, worst 0.43 |
| Nystrom landmark extension | ties k-means at 5x the cost | needs 256 landmarks and 10.6 ms to reach 82.90 oracle |
| `torch.lobpcg`, batched dense | 189 ms/shape | slower than the full decomposition |
| multi-curve kNN | -0.9 oracle, -10 boundary recall | exact kNN is already 0.19 ms at 2048 points |
| boundary refinement on a spectral embedding | +0.04 oracle | it only pays on k-means, where it is the concavity cue's only route in |
| Zelnik-Manor local scaling | ~0 | |
| VCCS supervoxels | -0.5 class-mIoU at 14x the cost | best boundary recall here, but lower oracle at matched regions, and CPU-bound |
| VCCS boundary-aware BFS | 81.03 vs 81.37 oracle (CPU), 83.57 vs 83.29 (GPU) | within noise either way |
| more landmarks (512) | worse everywhere | |

## Reproducing

```bash
bash run_final.sh        # the headline table
bash run_default.sh      # the shipped configuration
bash run_solver.sh       # sparse vs k-means on the end task
bash run_layout.sh       # the feature-layout ablation
bash run_vccs.sh         # the VCCS partition, end task and ceiling
python probe_superpoints.py --n_sp 32 48 64 96 128   # partition quality at matched counts
python bench_part.py --cls chair                     # stage-by-stage timing
```
