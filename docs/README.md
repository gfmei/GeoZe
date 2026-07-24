# Interactive scan viewer (GitHub Pages)

**Live at [gfmei.github.io/GeoZe](https://gfmei.github.io/GeoZe/)**

A self-contained WebGL viewer for SemGeoZe v2 results — no server, no GPU, no dependencies, nothing
to keep alive. Everything is precomputed, so it keeps working long after any Space goes to sleep.

```
docs/
  index.html            the whole viewer (WebGL2 point renderer, no external libraries)
  scenes/manifest.json  which scenes ship, with their honest per-scene mIoU
  scenes/*.json         base64 typed arrays: xyz, rgb, gt, point, meanpool, semgeozev2, region
```

## Using it

Pick a scene from the dropdown, then switch colour mode with the chips:

| mode | what you are looking at |
|---|---|
| **scan RGB** | the raw RGB-D scan, no predictions — the input |
| **ground truth** | ScanNet's 20-class annotation |
| **per-point VLM** | classifying each point's fused LSeg feature directly (0.4972 mIoU on full val) |
| **region mean pooling** | the baseline: average the features inside each segment, then classify (0.5564) |
| **SemGeoZe v2** | ours: merge regions first, then pool and classify (0.5606) |
| **merged regions** | the partition itself — colour is region identity, so you can see what merged |
| **errors** | SemGeoZe v2 correct (green) vs wrong (red) vs unlabelled (grey) |

Controls: **drag** to orbit, **wheel** to zoom, **shift-drag** to pan.

The instructive comparison is *per-point VLM* → *region mean pooling*: most of the jump comes from
pooling at all. Then *region mean pooling* → *merged regions* shows what SemGeoZe v2 adds — segments
coalescing into larger, still-pure regions.

The strip under the viewport reports that scene's own mIoU for all three methods, and the delta is
signed: it is **negative on some scenes**, which is the honest picture (see below).

## Deploying

```bash
git add docs && git commit -m "Add interactive scan viewer" && git push
```

Then on GitHub: **Settings → Pages → Source: `Deploy from a branch` → `main` / `/docs` → Save.**
The page appears at `https://<user>.github.io/<repo>/` within a minute or two.

To check it locally first — a plain `file://` open will **not** work, because the scene files are
fetched:

```bash
python -m http.server 8000 --directory docs
# then open http://localhost:8000
```

## Regenerating the scenes

```bash
python semseg/export_demo.py --candidates 40                 # writes scenes/*.json + manifest.json
python semseg/export_demo.py --candidates 40 --voxel 0.05    # coarser, smaller payload
```

The exporter scores candidate val scenes and picks one from each of the **neutral / typical /
good** bands of the per-scene SemGeoZe v2-minus-mean-pooling distribution — deliberately *not* the
three most favourable ones — and writes each scene's own mIoU into the manifest. Currently:

| scene | band | gain vs mean pooling |
|---|---|---|
| `scene0606_01` | neutral | −0.0395 |
| `scene0207_00` | typical | −0.0031 |
| `scene0583_02` | good | +0.0215 |

Per-scene mIoU moves in both directions even though the full-val aggregate is +0.43, because
aggregate mIoU is not the mean of per-scene mIoUs — rare classes dominate it. Keeping a losing
scene in the demo is deliberate: a viewer who clicks through should see the same picture the paper
reports.

`--voxel` controls the browser downsample (default 3.5 cm → 30–95k points, under 2 MB per scene).
Every number shown is computed on the **full-resolution** scan regardless.

## Adding your own scene

Export it, then append an entry to `scenes/manifest.json`:

```json
{"scene": "scene0011_00", "band": "custom", "n_classes": 11,
 "miou": {"point": 0.41, "meanpool": 0.52, "semgeozev2": 0.53},
 "gain": 0.01, "points": 48000, "file": "scene0011_00.json"}
```

## Notes

- Needs **WebGL2**; the page says so plainly if the browser lacks it.
- Light and dark themes both supported, following the OS preference.
- The viewer also runs fully offline if the scenes are inlined as `window.__SCENES__ =
  {manifest, data}` before the main script — that is how the shareable single-file build is made.
