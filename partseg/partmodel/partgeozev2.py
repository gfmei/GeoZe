"""PartGeoZe v2 — geometrically-driven aggregation for zero-shot 3D PART segmentation.

Object-level counterpart of semseg/semmodel/semgeozev2.py and the successor of partgeoze.py.
Same premise as GeoZe (use the shape's geometry, with no trainable parameter, to clean up a VLM's
per-point features before matching them to text) and the same design rule as SemGeoZe v2:

    aggregation may ENLARGE a region only when the enlargement is structurally justified, and it
    must never average across regions that stay separate.

    spectral superpoints     partmodel/spectral.py — XYZ + normals + FPFH, never the VLM feature
      -> region mean pooling z_m
      -> HierMerge           adjacency-constrained mutual-best-match agglomeration (the scene
                             module, unchanged, run on the flattened batch)
      -> InterStructuralAttn optional boundary-gated residual, off by default
      -> per-point recovery  f' = Norm(f + a (z_r(i) - f))

What is different from the scene version, and why:

  * The partition is spectral clustering rather than VCCS / mesh segments.  A shape is a single
    closed surface with sharp part boundaries (seat / leg, wing / body), which is exactly what a
    normalised-cut on a normal+FPFH-weighted graph finds; at 2k points the eigendecomposition is
    a single batched call.
  * Merge decisions can use CENTRED features (`center=True`).  Multi-view CLIP features of depth
    maps share a large common component, so raw cosines between adjacent regions of *different*
    parts are still ~0.9; subtracting the shape's mean feature before the cosine makes th_f a
    threshold on what distinguishes parts rather than on what every region shares.  The pooled
    feature that is classified is always the uncentred one.
  * Everything is batched: B shapes are flattened into one graph of B*N points with no edge
    between shapes, so the flat scatter-reduces of semseg.semmodel apply as they are.

GeoZe builds a knn patch around every superpoint and runs Sinkhorn attention inside each; here
every stage is a segment reduction over an edge list plus one eigendecomposition, which is what
makes it cheap (see part_run.py --model v2 for the timing next to GeoZe on the same GPU).
"""
import torch
import torch.nn.functional as F
from torch import nn

from partseg.partmodel.spectral import flat_pairs, hist_embed, knn_idx, superpoints
from semseg.semmodel.common import seg_mean
from semseg.semmodel.semgeozev2 import HierMerge, InterStructuralAttn


class PartGeoZeV2(nn.Module):
    """Training-free refinement of per-point VLM features on a batch of shapes.

    forward(xyz, normals, fpfhs, feats, seg=None)
        -> (refined feats [B,N,D], seg [B,N], reg [B,N], region feats [R,D] or None)

    `seg` is the superpoint partition, `reg` the merged region every point finally pooled over.
    Pass `seg` (flat [B*N] ids, contiguous and grouped by shape) to reuse a partition.
    """

    def __init__(self, n_sp=64, knn=10, th_f=0.5, th_n=0.3, rounds=0, part='spectral',
                 center=True, alpha=1.0, gamma0=0.0, w_x=1.0, w_n=1.0, w_g=1.0, w_v=0.0,
                 self_tune=False, n_ev=0, embed='dense', n_land=256, ortho=False,
                 land='curve', seed='curve', sparse_iters=50, vccs_voxel=0.05,
                 vccs_w_s=0.4, vccs_w_f=1.0, vccs_seed='fps', vccs_boundary=0.0,
                 refine=0, min_size=4,
                 split=True, eig_dtype=torch.float32):
        super().__init__()
        self.n_sp, self.knn, self.part, self.center, self.alpha = n_sp, knn, part, center, alpha
        self.w, self.min_size, self.split, self.eig_dtype = (w_x, w_n, w_g), min_size, split, eig_dtype
        self.w_v, self.self_tune, self.n_ev = w_v, self_tune, n_ev
        self.embed, self.n_land, self.refine = embed, n_land, refine
        self.ortho, self.land, self.seed = ortho, land, seed
        self.sparse_iters = sparse_iters
        self.vccs = dict(vccs_voxel=vccs_voxel, vccs_w_s=vccs_w_s, vccs_w_f=vccs_w_f,
                         vccs_seed=vccs_seed, vccs_boundary=vccs_boundary)
        self.hier = HierMerge(rounds, th_f, th_n)
        self.gattn = InterStructuralAttn(gamma0, colour_gate=False)

    def graph(self, xyz):
        return knn_idx(xyz, self.knn)

    def partition(self, xyz, nrm, gfe, idx):
        return superpoints(xyz, nrm, gfe, idx, n_sp=self.n_sp, method=self.part,
                           w_x=self.w[0], w_n=self.w[1], w_g=self.w[2], w_v=self.w_v,
                           self_tune=self.self_tune, n_ev=self.n_ev,
                           embed=self.embed, n_land=self.n_land, refine=self.refine,
                           ortho=self.ortho, land=self.land, seed=self.seed,
                           sparse_iters=self.sparse_iters, **self.vccs,
                           split=self.split,
                           min_size=self.min_size, eig_dtype=self.eig_dtype)

    def merge_feats(self, fn, B, N):
        """The feature the merge criterion sees: per-shape mean-centred when `center` is on."""
        if not self.center:
            return fn
        f = fn.view(B, N, -1)
        return (f - f.mean(1, keepdim=True)).reshape(B * N, -1)

    def forward(self, xyz, normals, fpfhs, feats, seg=None):
        B, N, D = feats.shape
        nrm = F.normalize(normals, dim=-1)
        gfe = hist_embed(fpfhs)
        idx = self.graph(xyz)
        i, j = flat_pairs(idx)
        if seg is None:
            seg = self.partition(xyz, nrm, gfe, idx)
        S = int(seg.max()) + 1

        f = feats.reshape(B * N, D)
        valid = f.norm(dim=-1) > 0                      # never-seen points carry a zero feature
        fn = F.normalize(f, dim=-1)
        w = valid.to(fn.dtype)
        nrm_f, gfe_f = nrm.reshape(B * N, 3), gfe.reshape(B * N, -1)

        # --- hierarchical merging: enlarge the pooling support ---
        par = self.hier(nrm_f, self.merge_feats(fn, B, N), valid, seg, S, i, j)
        reg = par[seg]
        R = int(par.max()) + 1
        z = F.normalize(seg_mean(fn, reg, R, w=w), dim=-1)

        # --- optional inter-region context, gated by the geometric boundary ---
        if self.gattn.gamma0 > 0:
            edges = self.gattn.region_graph(nrm_f, reg, R, i, j)
            z2, gamma = self.gattn(xyz.new_zeros(B * N, 3), nrm_f, gfe_f, reg, R, z, edges)
        else:
            z2, gamma = z, z.new_zeros(R)

        # --- per-point recovery ---
        # With alpha=1 and the inter-region residual off, this collapses to out = z[reg]: the
        # point never needs its own vector, so the caller can classify the R region vectors and
        # propagate the LABEL instead of broadcasting a 512-d feature to every point.  `zr` is
        # that region feature when the collapse is exact, and None when it is not.
        a = torch.where(valid, torch.full_like(w, self.alpha), torch.ones_like(w)).unsqueeze(1)
        b = a * (gamma[reg] > 0).to(a.dtype).unsqueeze(1)
        out = fn + a * (z[reg] - fn) + b * (z2[reg] - z[reg])
        zr = z if (self.alpha == 1.0 and self.gattn.gamma0 == 0) else None
        return F.normalize(out, dim=-1).view(B, N, D), seg.view(B, N), reg.view(B, N), zr
