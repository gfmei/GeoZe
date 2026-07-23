"""
vccs.py — Voxel Cloud Connectivity Segmentation (vectorized)

Vendored and trimmed: the FPFH helpers, the label-smoothing utility and the demo/test drivers
of the original were dropped because nothing in this repo calls them (FPFH here comes from
open3d in sem_prep.py). The VCCS class itself is unmodified.
============================================================
Python re-implementation of:
    J. Papon, A. Abramov, M. Schoeler, F. Wörgötter,
    "Voxel Cloud Connectivity Segmentation — Supervoxels for Point Clouds",
    CVPR 2013.

Same algorithm as the reference version; the Python-level for-loops have been
replaced with vectorized numpy ops:

  • `build_adjacency` -- list-of-lists + per-voxel × 26-offset loop
    →  CSR adjacency (`adj_starts`, `adj_data`) built with vectorized
       `np.searchsorted` over packed integer keys. Two passes (count + fill)
       so peak memory stays O(m), not O(26·m).

  • `_grow` (BFS) -- per-frontier-voxel python loop + per-claim dict resolution
    →  fully vectorized per layer. We build a flat (src,dst) edge array for the
       entire frontier via a CSR ragged-range gather, score all edges in one
       pass, and resolve concurrent claims with `lexsort` + first-occurrence
       pick. Memory bounded by O(edges in current layer), which is what the
       old `claims` dict already held.

  • `_update_seeds` -- per-cluster loop with `np.where` for members
    →  `bincount` for cluster sums + `lexsort` for argmin per cluster.

  • `_seed` refinement -- per-seed × per-candidate variance loop
    →  per-voxel "normal variance" pre-computed once via CSR scatter-add,
       then a single lexsort to pick min-score voxel per seed.

The semantics are unchanged. The flow-constrained nearest-seed assignment
still respects voxel adjacency; the combined distance is still Eq. 1.

Dependencies: numpy, scipy.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


# ─────────────────────────  colour utility  ─────────────────────────

def rgb_to_lab(rgb):
    """sRGB in [0,1] (N,3) → CIELab (N,3). D65 reference white."""
    a = 0.055
    mask = rgb > 0.04045
    lin = np.where(mask, ((rgb + a) / (1 + a)) ** 2.4, rgb / 12.92)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T
    ref = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / ref
    d = 6 / 29
    f = np.where(xyz > d ** 3, xyz ** (1 / 3), xyz / (3 * d * d) + 4 / 29)
    l = 116.0 * f[:, 1] - 16.0
    a_ = 500.0 * (f[:, 0] - f[:, 1])
    b_ = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([l, a_, b_], axis=1)


# ─────────────────────────  normals  ─────────────────────────

def estimate_normals(pts, k=20):
    """Per-point normals via local PCA. Vectorized over all points."""
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k)
    nb = pts[idx]                          # (n, k, 3)
    c = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / (k - 1)
    _, v = np.linalg.eigh(cov)
    return v[:, :, 0]                      # smallest eigenvector per point


def normal_distance(n_a, n_b):
    """Angular distance for unit normals, in [0, 1]. Flip-invariant.

    Used for the geometric term D_f of Eq. 1.
    """
    return 1.0 - np.abs(np.sum(n_a * n_b, axis=-1))


# ─────────────────────────  Z-curve (Morton) ordering  ─────────────────────────

def _zcurve_argsort(pts: np.ndarray, depth: int = 10) -> np.ndarray:
    """
    Sort 3D points by Z-curve (Morton) code for spatial locality.

    Normalises pts to a [0, 2^depth - 1]³ integer grid, then bit-interleaves
    the three coordinates into a single int64 Morton code.  Points with nearby
    codes are guaranteed to be spatially close (same locality-preservation
    guarantee as the Hilbert curve, simpler to compute without extra libraries).

    Parameters
    ----------
    pts   : (N, 3) float — 3D coordinates, any scale
    depth : int          — bits per axis; depth=10 → 1024 cells per axis.
                           Keep depth ≤ 20 to stay within int64.

    Returns
    -------
    (N,) int64 — argsort indices that reorder pts in Z-curve order
    """
    mn  = pts.min(axis=0)
    mx  = pts.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    grid = np.clip(
        np.floor((pts - mn) / span * (2 ** depth - 1)).astype(np.int64),
        0, 2 ** depth - 1,
    )
    x, y, z = grid[:, 0], grid[:, 1], grid[:, 2]
    code = np.zeros(len(pts), dtype=np.int64)
    for bit in range(depth):
        mask = np.int64(1 << bit)
        code |= (
            ((x & mask) << (2 * bit))
            | ((y & mask) << (2 * bit + 1))
            | ((z & mask) << (2 * bit + 2))
        )
    return np.argsort(code, kind="stable")


# ─────────────────────────  voxelization  ─────────────────────────

def voxelize(pts, col, nrm, voxel_res):
    """Average points / colours / normals into voxels of size voxel_res."""
    keys = np.floor(pts / voxel_res).astype(np.int64)
    unq, inv = np.unique(keys, axis=0, return_inverse=True)
    m = len(unq)
    cnt = np.bincount(inv, minlength=m).astype(np.float64)

    def _mean(x):
        # bincount per channel is faster than np.add.at on a 2D buffer.
        out = np.empty((m, x.shape[1]))
        for d in range(x.shape[1]):
            out[:, d] = np.bincount(inv, weights=x[:, d], minlength=m) / cnt
        return out

    v_pts = _mean(pts)
    v_col = _mean(col) if col is not None else None
    v_nrm = None
    if nrm is not None:
        v_nrm = _mean(nrm)
        v_nrm = v_nrm / np.clip(np.linalg.norm(v_nrm, axis=1, keepdims=True), 1e-8, None)

    return unq, inv, v_pts, v_col, v_nrm


# ───────────────────  CSR 26-connectivity adjacency  ───────────────────

def build_adjacency_csr(keys):
    """26-connectivity adjacency as a CSR graph.

    Returns:
        adj_starts : (m+1,) int64 — offsets into adj_data
        adj_data   : (E,)   int64 — destination voxel ids

    Use `adj_data[adj_starts[v]:adj_starts[v+1]]` for the 1-ring of voxel v.

    Two passes (count + fill) keep peak memory at O(m) per offset instead of
    materialising 26 (m,)-sized edge buffers simultaneously.
    """
    m = len(keys)
    mn = keys.min(axis=0)
    shifted = (keys - mn).astype(np.int64)    # all >= 0
    base = np.int64(int(shifted.max()) + 2)   # +2 keeps ±1 offsets in range

    def pack(k):
        return (k[:, 0] * base + k[:, 1]) * base + k[:, 2]

    packed = pack(shifted)
    order = np.argsort(packed, kind="stable")
    packed_sorted = packed[order]

    offsets = np.array([(i, j, k)
                        for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                        if not (i == 0 and j == 0 and k == 0)],
                       dtype=np.int64)

    # ── pass 1 : count neighbours per source
    counts = np.zeros(m, dtype=np.int64)
    for off in offsets:
        packed_off = pack(shifted + off)
        pos = np.searchsorted(packed_sorted, packed_off)
        pos_safe = np.clip(pos, 0, m - 1)
        match = packed_sorted[pos_safe] == packed_off
        counts += match.astype(np.int64)

    adj_starts = np.empty(m + 1, dtype=np.int64)
    adj_starts[0] = 0
    np.cumsum(counts, out=adj_starts[1:])
    adj_data = np.empty(int(adj_starts[-1]), dtype=np.int64)

    # ── pass 2 : fill in adj_data via per-source fill pointer
    fill = adj_starts[:-1].copy()
    for off in offsets:
        packed_off = pack(shifted + off)
        pos = np.searchsorted(packed_sorted, packed_off)
        pos_safe = np.clip(pos, 0, m - 1)
        match = packed_sorted[pos_safe] == packed_off
        src = np.where(match)[0]
        dst = order[pos_safe[src]]
        adj_data[fill[src]] = dst
        fill[src] += 1

    return adj_starts, adj_data


def _csr_gather(frontier, adj_starts, adj_data):
    """Flatten the 1-rings of every voxel in `frontier` into a parallel
    (src_id, dst) pair of arrays. src_id is the *index into frontier*, not
    the voxel id, so the caller does `frontier[src_id]` to recover voxels.

    Memory: O(total neighbours of the frontier).
    """
    starts = adj_starts[frontier]
    counts = (adj_starts[frontier + 1] - starts).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64))
    src_id = np.repeat(np.arange(len(frontier), dtype=np.int64), counts)
    cumstart = np.empty_like(counts)
    cumstart[0] = 0
    cumstart[1:] = np.cumsum(counts[:-1])
    local = np.arange(total, dtype=np.int64) - np.repeat(cumstart, counts)
    dst = adj_data[np.repeat(starts, counts) + local]
    return src_id, dst


# ─────────────────────────  VCCS  ─────────────────────────

@dataclass
class VccsParams:
    voxel_res:           float = 0.008   # R_voxel
    seed_res:            float = 0.10    # R_seed (grid seeding only)
    w_c:                 float = 0.20    # colour weight
    w_s:                 float = 0.40    # spatial weight
    w_f:                 float = 1.00    # normal weight (D_f term)
    max_iter:            int   = 10
    color:               bool  = True
    normal:              bool  = True
    search_radius_mult:  float = 1.8     # BFS cap in units of R_seed
    seed_filter_mult:    float = 0.5     # drop seeds farther than this · R_seed
    n_superpoints:       int   = 0       # >0 → target seed count
    curve_depth:         int   = 10      # bits per axis for Z-curve encoding
    # Seeding strategy
    # "auto"   : grid when n_superpoints==0, zcurve otherwise (legacy behavior)
    # "fps"    : farthest-point sampling — best coverage across thin parts
    # "zcurve" : Z-curve stride (fast, less uniform than fps)
    # "grid"   : regular 3D grid (original VCCS)
    seed_mode:           str   = "auto"
    # Spatial distance in BFS
    # False: Euclidean distance from seed center (standard SLIC-style)
    # True : BFS hop count / max_layers — topology-aware, respects thin parts
    use_geodesic:        bool  = False
    # Boundary-aware BFS: add `boundary_weight` extra cost when crossing an edge
    # where |n_src · n_dst| < boundary_thresh.  Soft barrier at part transitions.
    boundary_weight:     float = 0.0
    boundary_thresh:     float = 0.5


class VCCS:
    """Voxel Cloud Connectivity Segmentation — vectorized."""

    def __init__(self, params=None):
        self.p = params or VccsParams()

    # ........................................................  public  ........

    def fit(self, pts, col=None, nrm=None):
        p = self.p
        if col is not None and col.max() > 1.5:
            col = col.astype(np.float64) / 255.0
        if nrm is None and p.normal:
            nrm = estimate_normals(pts, k=20)

        # 1. voxelize ─────────────────────────────────────────────
        keys, inv, v_pts, v_col, v_nrm = voxelize(pts, col, nrm, p.voxel_res)
        v_lab = rgb_to_lab(v_col) if (p.color and v_col is not None) else None

        # 2. adjacency (CSR) ──────────────────────────────────────
        adj_starts, adj_data = build_adjacency_csr(keys)

        # 3. seed placement ───────────────────────────────────────
        seed_idx, seed_res = self._seed(v_pts, v_nrm, adj_starts, adj_data)

        # 4. iterative clustering ─────────────────────────────────
        labels = -np.ones(len(v_pts), dtype=np.int64)
        for it in range(p.max_iter):
            new_labels = self._grow(seed_idx, v_pts, v_lab, v_nrm,
                                    adj_starts, adj_data, seed_res)
            new_seeds = self._update_seeds(seed_idx, new_labels, v_pts)
            converged = (np.array_equal(new_seeds, seed_idx)
                         and np.array_equal(new_labels, labels))
            labels = new_labels
            seed_idx = new_seeds
            if converged:
                break

        # 5. map back to points
        return labels[inv], labels, v_pts, seed_idx

    # ........................................................  internals  ....

    def _refine_seeds(self, seeds, v_pts, v_nrm, adj_starts, adj_data):
        """Move each seed to the lowest-normal-gradient voxel in its 1-ring."""
        m = len(v_pts)
        cnt = (adj_starts[1:] - adj_starts[:-1]).astype(np.float64)
        src = np.repeat(np.arange(m, dtype=np.int64), cnt.astype(np.int64))
        nd = v_nrm[adj_data]
        sum_n  = np.empty((m, 3))
        sum_n2 = np.empty((m, 3))
        for d_ in range(3):
            sum_n[:, d_]  = np.bincount(src, weights=nd[:, d_], minlength=m)
            sum_n2[:, d_] = np.bincount(src, weights=nd[:, d_] * nd[:, d_], minlength=m)
        safe   = np.maximum(cnt, 1.0)[:, None]
        mean_n = sum_n / safe
        score  = (sum_n2 / safe - mean_n * mean_n).sum(axis=1)
        score[cnt == 0] = np.inf

        seed_st = adj_starts[seeds]
        seed_cn = (adj_starts[seeds + 1] - seed_st).astype(np.int64)
        n_seeds = len(seeds)
        cand_seed_self = np.arange(n_seeds, dtype=np.int64)
        cand_vox_self  = seeds

        total_nb = int(seed_cn.sum())
        if total_nb > 0:
            cand_seed_nb = np.repeat(cand_seed_self, seed_cn)
            cumstart = np.empty_like(seed_cn)
            cumstart[0] = 0
            cumstart[1:] = np.cumsum(seed_cn[:-1])
            local = np.arange(total_nb, dtype=np.int64) - np.repeat(cumstart, seed_cn)
            cand_vox_nb = adj_data[np.repeat(seed_st, seed_cn) + local]
        else:
            cand_seed_nb = np.empty(0, dtype=np.int64)
            cand_vox_nb  = np.empty(0, dtype=np.int64)

        cand_seed  = np.concatenate([cand_seed_self, cand_seed_nb])
        cand_vox   = np.concatenate([cand_vox_self,  cand_vox_nb])
        cand_score = score[cand_vox]
        order = np.lexsort((cand_score, cand_seed))
        _, first = np.unique(cand_seed[order], return_index=True)
        return cand_vox[order][first].astype(np.int64)

    def _grid_seed(self, v_pts, v_nrm, adj_starts, adj_data):
        """Grid seeding — snap to nearest voxel, then refine. Returns (seeds, seed_res)."""
        p = self.p
        mn, mx = v_pts.min(axis=0), v_pts.max(axis=0)
        gx = np.arange(mn[0], mx[0] + p.seed_res, p.seed_res)
        gy = np.arange(mn[1], mx[1] + p.seed_res, p.seed_res)
        gz = np.arange(mn[2], mx[2] + p.seed_res, p.seed_res)
        grid = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)

        tree = cKDTree(v_pts)
        d, idx = tree.query(grid, k=1)
        keep  = d < p.seed_res * p.seed_filter_mult
        seeds = np.unique(idx[keep]).astype(np.int64)

        if v_nrm is not None and len(seeds) > 0:
            seeds = self._refine_seeds(seeds, v_pts, v_nrm, adj_starts, adj_data)

        return seeds, p.seed_res

    def _hilbert_seed(self, v_pts, v_nrm, adj_starts, adj_data):
        """Z-curve stride seeding — targets p.n_superpoints seeds. Returns (seeds, effective_seed_res)."""
        p   = self.p
        m   = len(v_pts)
        n   = min(p.n_superpoints, m)

        order  = _zcurve_argsort(v_pts, depth=p.curve_depth)
        stride = max(1, m // n)
        seeds  = order[np.arange(0, m, stride, dtype=np.int64)[:n]].astype(np.int64)

        mn, mx = v_pts.min(axis=0), v_pts.max(axis=0)
        vol    = float(np.prod(np.maximum(mx - mn, 1e-6)))
        eff_sr = max(p.voxel_res, (vol / max(n, 1)) ** (1.0 / 3.0))

        if v_nrm is not None and len(seeds) > 0:
            seeds = self._refine_seeds(seeds, v_pts, v_nrm, adj_starts, adj_data)

        return seeds, eff_sr

    def _fps_seed(self, v_pts, v_nrm, adj_starts, adj_data):
        """Farthest-point sampling seeding — targets p.n_superpoints seeds.

        FPS guarantees that every seed is as far as possible from all previously
        chosen seeds, giving the most uniform coverage across thin and concave
        parts.  Cost is O(n · m) where n = n_superpoints, m = voxel count —
        typically a few ms for ShapeNet-scale clouds.

        Returns (seeds, effective_seed_res).
        """
        p = self.p
        m = len(v_pts)
        n = min(max(1, p.n_superpoints), m)

        min_dists = np.full(m, np.inf)
        seeds = np.empty(n, dtype=np.int64)
        seeds[0] = 0
        d = np.linalg.norm(v_pts - v_pts[0], axis=1)
        np.minimum(min_dists, d, out=min_dists)

        for i in range(1, n):
            seeds[i] = int(np.argmax(min_dists))
            d = np.linalg.norm(v_pts - v_pts[seeds[i]], axis=1)
            np.minimum(min_dists, d, out=min_dists)

        # Effective seed resolution = mean nearest-neighbor distance among seeds.
        if n > 1:
            s_pts = v_pts[seeds]
            tree  = cKDTree(s_pts)
            d_nn, _ = tree.query(s_pts, k=2)   # k=2: self + nearest other seed
            eff_sr = max(p.voxel_res, float(d_nn[:, 1].mean()))
        else:
            eff_sr = p.voxel_res

        if v_nrm is not None and len(seeds) > 0:
            seeds = self._refine_seeds(seeds, v_pts, v_nrm, adj_starts, adj_data)

        return seeds, eff_sr

    def _seed(self, v_pts, v_nrm, adj_starts, adj_data):
        """Route to the requested seeding strategy. Returns (seed_idx, seed_res)."""
        mode = self.p.seed_mode
        if mode == "auto":
            if self.p.n_superpoints > 0:
                return self._hilbert_seed(v_pts, v_nrm, adj_starts, adj_data)
            return self._grid_seed(v_pts, v_nrm, adj_starts, adj_data)
        if mode == "fps":
            return self._fps_seed(v_pts, v_nrm, adj_starts, adj_data)
        if mode == "zcurve":
            return self._hilbert_seed(v_pts, v_nrm, adj_starts, adj_data)
        if mode == "grid":
            return self._grid_seed(v_pts, v_nrm, adj_starts, adj_data)
        raise ValueError(f"Unknown seed_mode={mode!r}; choose 'auto', 'fps', 'zcurve', or 'grid'")

    def _grow(self, seed_idx, v_pts, v_lab, v_nrm, adj_starts, adj_data, seed_res):
        """Flow-constrained nearest-seed BFS — fully vectorized per layer.

        Per layer:
          1. gather the entire frontier's 1-rings into a flat (src, dst) edge
             array via the CSR ragged-range helper;
          2. compute Eq.1 distances for every edge in one numpy pass;
          3. resolve concurrent claims (multiple seeds proposing the same dst)
             with lexsort + first-occurrence-per-dst — equivalent to
             min-d² wins;
          4. accept claims that beat the per-voxel `best_d`; winners form the
             next frontier.

        Memory: O(edges in current frontier) — the same upper bound as the
        old `claims` dict.
        """
        p = self.p
        m = len(v_pts)

        labels = -np.ones(m, dtype=np.int64)
        best_d = np.full(m, np.inf)
        labels[seed_idx] = np.arange(len(seed_idx), dtype=np.int64)
        best_d[seed_idx] = 0.0

        s_pts = v_pts[seed_idx]
        s_lab = v_lab[seed_idx] if v_lab is not None else None
        s_nrm = v_nrm[seed_idx] if v_nrm is not None else None

        max_layers = max(1, int(p.search_radius_mult * seed_res / p.voxel_res))
        spatial_norm = np.sqrt(3.0) * seed_res
        color_norm = 100.0

        frontier = seed_idx.copy()
        for layer_count in range(max_layers):
            if len(frontier) == 0:
                break

            src_id, dst = _csr_gather(frontier, adj_starts, adj_data)
            if dst.size == 0:
                break

            src_vox = frontier[src_id]
            k = labels[src_vox]                            # seed id per edge

            # Spatial term: geodesic (hop fraction) or Euclidean from seed center.
            if p.use_geodesic:
                geo_d = (layer_count + 1.0) / max(max_layers, 1)
                d2 = p.w_s * geo_d * geo_d * np.ones(len(dst))
            else:
                ds = np.linalg.norm(v_pts[dst] - s_pts[k], axis=1) / spatial_norm
                d2 = p.w_s * ds * ds

            if s_lab is not None:
                dc = np.linalg.norm(v_lab[dst] - s_lab[k], axis=1) / color_norm
                d2 = d2 + p.w_c * dc * dc
            if s_nrm is not None:
                df = normal_distance(v_nrm[dst], s_nrm[k])
                d2 = d2 + p.w_f * df * df

            # Boundary-aware cost: penalise edges where adjacent normals diverge.
            # High-curvature edges are soft barriers that the BFS avoids crossing.
            if p.boundary_weight > 0.0 and v_nrm is not None:
                src_n = v_nrm[src_vox]
                dst_n = v_nrm[dst]
                compat = np.abs(np.einsum("ei,ei->e", src_n, dst_n))
                d2 = d2 + p.boundary_weight * np.maximum(0.0, p.boundary_thresh - compat)

            # Min d² per dst — sort by (dst, d2) then take first occurrence.
            order = np.lexsort((d2, dst))
            dst_s = dst[order]
            d2_s  = d2[order]
            k_s   = k[order]
            _, first = np.unique(dst_s, return_index=True)
            cand_dst, cand_d2, cand_k = dst_s[first], d2_s[first], k_s[first]

            accept = cand_d2 < best_d[cand_dst]
            won_dst = cand_dst[accept]
            labels[won_dst] = cand_k[accept]
            best_d[won_dst] = cand_d2[accept]

            frontier = won_dst

        return labels

    def _update_seeds(self, seed_idx, labels, v_pts):
        """Move each seed to the voxel nearest its cluster centroid —
        vectorized via bincount centroids + lexsort argmin."""
        n_seeds = len(seed_idx)
        valid = labels >= 0
        v_lbl = labels[valid]
        v_p   = v_pts[valid]
        v_ix  = np.where(valid)[0]

        cnt = np.bincount(v_lbl, minlength=n_seeds).astype(np.float64)
        cent = np.empty((n_seeds, 3))
        for d_ in range(3):
            cent[:, d_] = np.bincount(v_lbl, weights=v_p[:, d_], minlength=n_seeds)
        nz = cnt > 0
        cent[nz] /= cnt[nz, None]

        dist = np.linalg.norm(v_p - cent[v_lbl], axis=1)
        order = np.lexsort((dist, v_lbl))
        sorted_lbl = v_lbl[order]
        sorted_ix  = v_ix[order]
        uniq_lbl, first = np.unique(sorted_lbl, return_index=True)

        new = seed_idx.copy()
        new[uniq_lbl] = sorted_ix[first]
        return new
