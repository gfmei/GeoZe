"""SemGeoZeV2 — geometrically-driven aggregation for zero-shot 3D SCENE segmentation.

Scene counterpart of partseg/partmodel/partgeoze.py.  Same premise as GeoZe: use the point
cloud's geometry, with no trainable parameter anywhere, to clean up a VLM's per-point features
before they are matched against text.  The scene setting needs a different decomposition than
the object one, so the stages are:

    VCCS / mesh partition
      -> IntraConsensusAttn   attention ESTIMATES CONSENSUS inside a region (r_i), it does not
                              mix features; combined with a structural confidence q_i it gives
                              a robust pooled region vector z_m
      -> HierMerge            adjacency-constrained mutual-best-match agglomeration, which
                              ENLARGES the pooling support without ever averaging two regions
                              that the geometry keeps apart
      -> InterStructuralAttn  a boundary-gated residual that propagates context between regions
      -> per-point recovery

Measured contribution of each stage on ScanNet val (312 scenes, LSeg features, mesh segments,
against superpoint mean pooling at 0.5564 mIoU):

    IntraConsensusAttn   +0.0002    regions are already 98.5% homogeneous in VLM space, so
                                    reweighting their points cannot move the mean direction
    HierMerge            +0.0043    the only stage with a consistent effect
    InterStructuralAttn  +0.0015    within noise; its normal-based boundary gate reads 0.908
                                    on ScanNet, i.e. wide open (adjacent segments usually do
                                    have agreeing normals whether or not they are one object)

Intra and inter are therefore OFF by default in best_param and switchable from the command
line; do not re-enable them on ScanNet expecting a gain.  See semmodel/best_param.py.
"""
import torch
import torch.nn.functional as F
from torch import nn

from semseg.semmodel.common import EPS, robust_scale, seg_mean, seg_normal, seg_softmax  # noqa: F401


# --------------------------------------------------------------------------- #
#  Structural confidence  q_i                                                  #
# --------------------------------------------------------------------------- #
class StructuralConf(nn.Module):
    """q_i in (0,1]: how well point i fits the plane / normal / geometry / colour model of its
    own region.  Boundary points and partition errors go to 0."""

    def __init__(self, gamma=1.0, use_colour=True):
        super().__init__()
        self.gamma, self.use_colour = gamma, use_colour

    def forward(self, xyz, rgb, nrm, gfe, seg, S):
        nrm = F.normalize(nrm, dim=-1)                 # exact unit, else 1-|n.n| can go negative
        mu = seg_mean(xyz, seg, S)
        nbar = seg_normal(nrm, seg, S)
        gbar = F.normalize(seg_mean(gfe, seg, S), dim=-1)
        cbar = seg_mean(rgb, seg, S)

        d = [((xyz - mu[seg]) * nbar[seg]).sum(-1).abs(),                       # plane residual
             (1.0 - (nrm * nbar[seg]).sum(-1).abs()).clamp_min(0),              # normal deviation
             (1.0 - (F.normalize(gfe, dim=-1) * gbar[seg]).sum(-1)).clamp_min(0)]  # FPFH deviation
        if self.use_colour:
            d.append((rgb - cbar[seg]).norm(dim=-1))                            # colour deviation
        acc = torch.zeros_like(d[0])
        for dk in d:
            s = seg_mean(dk.unsqueeze(1), seg, S).squeeze(1)
            s = s.clamp_min(1e-2 * dk.mean()).clamp_min(EPS)   # region scale + global floor
            acc = acc + (dk / s[seg]).clamp(max=10.0)          # cap so one cue cannot veto
        return torch.exp(-self.gamma * acc / len(d))


# --------------------------------------------------------------------------- #
#  Intra-region consensus attention                                            #
# --------------------------------------------------------------------------- #
def _buckets(seg, valid, S, cap_min=32):
    """Group points by region into size-bucketed padded blocks.

    Inside a region the points are ordered valid-feature-first then random, so the first
    `n_anchor` slots are a random sample of the FEATURE-CARRYING points — exactly the landmark
    set the consensus estimator needs.  Yields (ids [B], idx [B,C], mask [B,C]).
    """
    N = seg.numel()
    dev = seg.device
    perm = torch.randperm(N, device=dev)
    key = seg[perm].to(torch.int64) * 2 + (~valid[perm]).to(torch.int64)
    order = perm[torch.argsort(key, stable=True)]
    counts = torch.bincount(seg, minlength=S)
    offsets = torch.cumsum(counts, 0) - counts
    cap, ar = cap_min, torch.arange(S, device=dev)
    while True:
        sel = (counts > (cap // 2 if cap > cap_min else 0)) & (counts <= cap)
        if sel.any():
            ids = ar[sel]
            base = offsets[ids].unsqueeze(1) + torch.arange(cap, device=dev).unsqueeze(0)
            mask = torch.arange(cap, device=dev).unsqueeze(0) < counts[ids].unsqueeze(1)
            yield ids, order[base.clamp(max=N - 1)] * mask, mask
        if cap >= int(counts.max()):
            break
        cap *= 2


class IntraConsensusAttn(nn.Module):
    """Parameter-free attention whose output is a per-point CONSENSUS SCORE, not a mixed feature.

    For anchors a and points i of the same region,

        e[a,i] = - sum_cue  w_cue * d_cue(a,i) / tau_cue(region)
        A[a,i] = softmax_i e[a,i]
        r_i    = mean_a A[a,i]                       <- column mass = support from the region

    r_i is never used to mix features (no A @ F); it only weights the pooling, together with the
    structural confidence q_i.  tau_cue is a per-region statistic of that cue's own distances, so
    every region gets its own temperature and there is nothing to tune.
    """

    def __init__(self, n_anchor=128, w_f=1.0, w_g=1.0, w_n=1.0, w_c=1.0, w_x=1.0,
                 scale='mean', use_q=True, blend='entropy', proto_round=True):
        super().__init__()
        self.n_anchor, self.scale, self.use_q = n_anchor, scale, use_q
        self.blend, self.proto_round = blend, proto_round
        self.w = dict(f=w_f, g=w_g, n=w_n, c=w_c, x=w_x)

    def forward(self, xyz, rgb, nrm, gfe, feat, valid, seg, S, q):
        N, D = feat.shape
        fn, gn = F.normalize(feat, dim=-1), F.normalize(gfe, dim=-1)
        nrm = F.normalize(nrm, dim=-1)
        r = torch.zeros(N, device=feat.device, dtype=feat.dtype)
        z_att = feat.new_zeros(S, D)
        eta = feat.new_zeros(S)

        for ids, idx, mask in _buckets(seg, valid, S):
            B, C = idx.shape
            A = min(self.n_anchor, C)
            pv = valid[idx] & mask
            av = mask[:, :A] & pv[:, :A]                       # anchors must carry a feature
            pair = av.unsqueeze(2) & pv.unsqueeze(1)           # [B,A,C] usable pairs
            pf, pg, pn = fn[idx], gn[idx], nrm[idx]
            px, pc = xyz[idx], rgb[idx]

            e = torch.zeros(B, A, C, device=feat.device, dtype=feat.dtype)

            def add(e, w, d):
                return e - w * (d / robust_scale(d, pair, self.scale)).clamp(max=50.0)

            if self.w['f']:
                e = add(e, self.w['f'], (1 - torch.bmm(pf[:, :A], pf.transpose(1, 2))).clamp_min(0))
            if self.w['g']:
                e = add(e, self.w['g'], (1 - torch.bmm(pg[:, :A], pg.transpose(1, 2))).clamp_min(0))
            if self.w['n']:
                e = add(e, self.w['n'], (1 - torch.bmm(pn[:, :A], pn.transpose(1, 2)).abs()).clamp_min(0))
            if self.w['c']:
                e = add(e, self.w['c'], torch.cdist(pc[:, :A], pc).pow(2))
            if self.w['x']:
                e = add(e, self.w['x'], torch.cdist(px[:, :A], px).pow(2))

            att = torch.softmax(e.masked_fill(~pair, -1e4), dim=-1) * pair.any(2, keepdim=True)
            ri = att.sum(1) / av.sum(1, keepdim=True).clamp_min(1)
            r[idx[mask]] = ri[mask]

            u = ri * (q[idx] if self.use_q else 1.0) * pv
            w = u / u.sum(1, keepdim=True).clamp_min(EPS)
            H = -(w * (w + EPS).log()).sum(1) / torch.log(pv.sum(1).clamp_min(2).float())
            za = torch.einsum('bc,bcd->bd', w.to(feat.dtype), fn[idx])
            if self.proto_round:                     # consensus discovery -> consensus verification
                p = F.normalize(za, dim=-1)
                sim = torch.einsum('bd,bcd->bc', p, fn[idx])
                tau = ((1 - sim) * pv).sum(1) / pv.sum(1).clamp_min(1)
                e2 = sim / tau.clamp_min(EPS).unsqueeze(1)
                if self.use_q:
                    e2 = e2 + (q[idx] + EPS).log()
                w2 = torch.softmax(e2.masked_fill(~pv, -1e4), dim=-1) * pv.any(1, keepdim=True)
                za = torch.einsum('bc,bcd->bd', w2.to(feat.dtype), fn[idx])
            z_att[ids] = F.normalize(za, dim=-1)
            eta[ids] = (1 - H).clamp(0, 1).to(feat.dtype)

        z_mean = F.normalize(seg_mean(fn, seg, S, w=valid.to(fn.dtype)), dim=-1)
        if self.blend == 'mean':
            z = z_mean
        elif self.blend == 'att':
            z = z_att
        else:                                        # entropy-gated: flat weights -> fall back to mean
            g = eta.unsqueeze(1)
            z = F.normalize((1 - g) * z_mean + g * z_att, dim=-1)
        return r, z, z_mean


# --------------------------------------------------------------------------- #
#  Hierarchical merging                                                        #
# --------------------------------------------------------------------------- #
class HierMerge(nn.Module):
    """Adjacency-constrained mutual-best-match agglomeration of regions.

    Pooling over a LARGER, semantically-correct region classifies far better than pooling over a
    150-point segment (pooling over the ground-truth region is worth +17 mIoU at the ceiling),
    but unconstrained merging destroys purity.  So merge only adjacent regions, and only in
    mutual-best-match pairs so nothing chains:

        admissible(m,n) iff cos(z_m,z_n) >= th_f and |n_m.n_n| >= th_n and b_mn >= th_n
        merge (m,n)     iff each is the other's best admissible neighbour

    b_mn is the mean |n_i.n_j| over the point pairs that actually straddle the two regions.
    """

    def __init__(self, rounds=10, th_f=0.85, th_n=0.30):
        super().__init__()
        self.rounds, self.th_f, self.th_n = rounds, th_f, th_n

    @staticmethod
    def _round(par, z, nbar, ms0, md0, bdot, th_f, th_n):
        S, dev = int(par.max()) + 1, z.device
        ms, md = par[ms0], par[md0]
        keep = ms != md
        if not keep.any():
            return par, False
        ms, md, bb = ms[keep], md[keep], bdot[keep]
        uk, inv = torch.unique(ms * S + md, return_inverse=True)
        cnt = torch.zeros(uk.numel(), device=dev).index_add_(0, inv, torch.ones_like(bb))
        bmean = torch.zeros(uk.numel(), device=dev).index_add_(0, inv, bb) / cnt.clamp_min(1)
        a, c = uk // S, uk % S
        s_f = (z[a] * z[c]).sum(-1)
        s_n = (nbar[a] * nbar[c]).sum(-1).abs()
        ok = (s_f >= th_f) & (s_n >= th_n) & (bmean >= th_n)
        if not ok.any():
            return par, False
        a, c, sc = a[ok], c[ok], s_f[ok]
        best = torch.full((S,), -1e9, device=dev).index_reduce_(0, a, sc, 'amax')
        bidx = torch.full((S,), -1, device=dev, dtype=torch.long)
        top = sc >= best[a] - 1e-6
        bidx[a[top]] = c[top]
        ar = torch.arange(S, device=dev)
        mutual = (bidx >= 0) & (bidx[bidx.clamp_min(0)] == ar)
        if not mutual.any():
            return par, False
        root = ar.clone()
        root[mutual] = torch.minimum(ar, bidx.clamp_min(0))[mutual]
        return torch.unique(root[par], return_inverse=True)[1], True

    def forward(self, nrm, feat, valid, seg, S, i, j):
        w = valid.to(feat.dtype)
        nn_ = F.normalize(nrm, dim=-1)
        bdot = (nn_[i] * nn_[j]).sum(-1).abs()
        par = torch.arange(S, device=feat.device)
        for _ in range(self.rounds):
            reg = par[seg]
            R = int(par.max()) + 1
            z = F.normalize(seg_mean(feat, reg, R, w=w), dim=-1)
            par, changed = self._round(par, z, seg_normal(nrm, reg, R), seg[i], seg[j],
                                       bdot, self.th_f, self.th_n)
            if not changed:
                break
        return par


# --------------------------------------------------------------------------- #
#  Inter-region structural attention                                           #
# --------------------------------------------------------------------------- #
class InterStructuralAttn(nn.Module):
    """z' = Norm(z + gamma_m * sum_n A_mn (z_n - z_m)) over the region graph.

    A RESIDUAL, not a replacement, so z stays the semantic anchor and attention only supplies a
    context correction.  log(b_mn) in the score is the boundary gate: a real geometric edge
    (wall vs cabinet) drives it towards 0 and switches the correction off.
    """

    def __init__(self, gamma0=0.5, knn_sp=12, boundary_gate=True, sigma_bc=0.25, colour_gate=True):
        super().__init__()
        self.gamma0, self.knn_sp = gamma0, knn_sp
        self.boundary_gate, self.sigma_bc, self.colour_gate = boundary_gate, sigma_bc, colour_gate

    def region_graph(self, nrm, seg, S, i, j):
        """Point kNN pairs -> region adjacency with the boundary gate measured on those pairs."""
        sm, sn = seg[i], seg[j]
        keep = sm != sn
        if not keep.any():
            return seg.new_zeros(0), seg.new_zeros(0), nrm.new_zeros(0)
        sm, sn, i, j = sm[keep], sn[keep], i[keep], j[keep]
        nn_ = F.normalize(nrm, dim=-1)
        dot = (nn_[i] * nn_[j]).sum(-1).abs()
        uk, inv = torch.unique(sm * S + sn, return_inverse=True)
        cnt = torch.zeros(uk.numel(), device=nrm.device).index_add_(0, inv, torch.ones_like(dot))
        acc = torch.zeros(uk.numel(), device=nrm.device).index_add_(0, inv, dot)
        src, dst, b = uk // S, uk % S, acc / cnt.clamp_min(1)
        if self.knn_sp > 0:                                   # cap degree by contact area
            o = torch.argsort(cnt, descending=True)
            o = o[torch.argsort(src[o], stable=True)]
            rank = torch.empty_like(o)
            rank[o] = torch.arange(o.numel(), device=o.device)
            first = torch.full((S,), o.numel(), dtype=torch.long, device=o.device)
            first = first.index_reduce_(0, src, rank, 'amin')
            sel = (rank - first[src]) < self.knn_sp
            src, dst, b = src[sel], dst[sel], b[sel]
        return src, dst, b

    def forward(self, rgb, nrm, gfe, seg, S, z, edges):
        src, dst, b_nrm = edges
        if src.numel() == 0 or self.gamma0 == 0:
            return z, z.new_zeros(S)
        cbar = seg_mean(rgb, seg, S)
        gbar = F.normalize(seg_mean(gfe, seg, S), dim=-1)
        nbar = seg_normal(nrm, seg, S)
        s_f = (z[src] * z[dst]).sum(-1)
        s_g = (gbar[src] * gbar[dst]).sum(-1)
        s_n = (nbar[src] * nbar[dst]).sum(-1).abs()

        def tau(v):
            return (1.0 - v).mean().clamp_min(EPS)

        e = s_f / tau(s_f) + s_g / tau(s_g) + s_n / tau(s_n)
        if self.boundary_gate:
            gate = b_nrm
            if self.colour_gate:
                gate = gate * torch.exp(-(cbar[src] - cbar[dst]).pow(2).sum(-1) / self.sigma_bc ** 2)
            e = e + (gate + EPS).log()
        a = seg_softmax(e, src, S)
        ent = torch.zeros(S, device=z.device, dtype=e.dtype).index_add_(0, src, -a * (a + EPS).log())
        deg = torch.zeros(S, device=z.device, dtype=e.dtype).index_add_(0, src, torch.ones_like(a))
        gamma = self.gamma0 * (1 - ent / torch.log(deg.clamp_min(2))).clamp(0, 1)
        delta = z.new_zeros(S, z.shape[1]).index_add_(0, src, a.unsqueeze(1) * (z[dst] - z[src]))
        return F.normalize(z + gamma.unsqueeze(1) * delta, dim=-1), gamma


# --------------------------------------------------------------------------- #
#  Full model                                                                  #
# --------------------------------------------------------------------------- #
class SemGeoZeV2(nn.Module):
    """Training-free scene-level refinement of per-point VLM features.

    forward(xyz, rgb, normals, fpfhs, feats, valid, seg, pairs) -> refined per-point feats.

    `seg`   [N] region id from the VCCS (or mesh) partition — built from XYZ+RGB+normals only,
            never from the VLM feature, so the partition can not inherit VLM noise.
    `pairs` (i, j) point-level neighbour pairs (multi-curve voting or KD-tree).
    """

    def __init__(self, n_anchor=128, q_gamma=1.0, th_f=0.85, th_n=0.30, rounds=10,
                 gamma0=0.0, use_intra=False, blend='entropy', alpha_min=1.0, w_c=1.0):
        super().__init__()
        self.use_intra, self.alpha_min, self.w_c = use_intra, alpha_min, w_c
        self.qconf = StructuralConf(q_gamma, use_colour=w_c > 0)
        self.iattn = IntraConsensusAttn(n_anchor=n_anchor, blend=blend, w_c=w_c)
        self.hier = HierMerge(rounds, th_f, th_n)
        self.gattn = InterStructuralAttn(gamma0, colour_gate=w_c > 0)

    def forward(self, xyz, rgb, normals, fpfhs, feats, valid, seg, pairs):
        S = int(seg.max().item()) + 1
        i, j = pairs
        if rgb is None:
            rgb = xyz.new_zeros(xyz.shape[0], 3)
        if fpfhs is None:                     # only the optional stages consume FPFH
            if self.use_intra or self.gattn.gamma0 > 0:
                raise ValueError('intra/inter attention need FPFH; pass fpfhs or disable them')
            fpfhs = xyz.new_zeros(xyz.shape[0], 1)
        fn = F.normalize(feats, dim=-1)
        w = valid.to(fn.dtype)

        # --- intra-region consensus pooling (or plain mean pooling) ---
        if self.use_intra:
            q = self.qconf(xyz, rgb, normals, fpfhs, seg, S)
            _, z, _ = self.iattn(xyz, rgb, normals, fpfhs, fn, valid, seg, S, q)
        else:
            q = torch.ones_like(xyz[:, 0])
            z = F.normalize(seg_mean(fn, seg, S, w=w), dim=-1)

        # --- hierarchical merging: enlarge the pooling support ---
        par = self.hier(normals, fn, valid, seg, S, i, j)
        reg = par[seg]
        R = int(par.max()) + 1
        z = F.normalize(seg_mean(fn, reg, R, w=w), dim=-1)

        # --- inter-region context, gated by the geometric boundary ---
        edges = self.gattn.region_graph(normals, reg, R, i, j)
        z2, gamma = self.gattn(rgb, normals, fpfhs, reg, R, z, edges)

        # --- per-point recovery: f' = Norm[f + a(z - f) + b(z' - z)] ---
        a = torch.full_like(q, self.alpha_min)
        if self.alpha_min < 1.0:
            qn = q / seg_mean(q.unsqueeze(1), reg, R).squeeze(1)[reg].clamp_min(EPS)
            a = self.alpha_min + (1 - self.alpha_min) * qn.clamp(0, 1)
        a = torch.where(valid, a, torch.ones_like(a))          # no VLM feature -> take z outright
        # gamma already scaled the residual inside z2; here it only decides how much of the
        # (already-scaled) correction reaches the point, so use the indicator, not gamma again.
        b = a * (gamma[reg] > 0).to(a.dtype)
        out = fn + a.unsqueeze(1) * (z[reg] - fn) + b.unsqueeze(1) * (z2[reg] - z[reg])
        return F.normalize(out, dim=-1), z, reg
