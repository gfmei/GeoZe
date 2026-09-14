"""Spectral superpoints for object point clouds — the partition stage of PartGeoZe v2.

An object has ~2k points, so everything here is dense and batched: a [B,N,N] affinity, one
batched eigendecomposition, and a batched k-means.  The partition is built from XYZ, normals
and FPFH only — never from the VLM feature — so the regions a feature is later pooled over
can not inherit that feature's noise (the same rule as sem_prep.py for scenes).

    affinity   w_ij = exp(-sum_cue d_cue(i,j) / tau_cue)   on the symmetric kNN graph
    embedding  top-K eigenvectors of D^-1/2 W D^-1/2, rows L2-normalised  (Ng-Jordan-Weiss)
    clusters   k-means in the embedding, seeded by farthest-point sampling
    cleanup    split every cluster into its connected components, absorb tiny fragments

tau_cue is the mean of that cue's distance over the shape's edges, so each cue lands on the same
scale and there is no per-dataset temperature to tune.  Two cheaper partitions (`kmeans` in the
concatenated cue space and plain `fps` Voronoi cells, which is what GeoZe's down_sample starts
from) are kept for the ablation in probe_partition.py.
"""
import torch
import torch.nn.functional as F

from libs.lib_utils import farthest_point_sample, index_points
from libs.serialization import encode

ORDERS = ('z', 'z-trans', 'hilbert', 'hilbert-trans')

EPS = 1e-8


def curve_sample(xyz, m, depth=10, order='hilbert'):
    """m well-spread point indices per shape, in one sort instead of m sequential passes.

    `farthest_point_sample` is a Python loop over the sample count: at m=256 that is ~1300 tiny
    GPU launches and it dominated the whole partition (25 of 26 ms/shape).  A space-filling
    curve gives the same "cover the shape evenly" property for one O(N log N) sort -- points
    adjacent on a Hilbert curve are spatially close, so a stride through the curve order is
    spread over the surface.  This is the sampler semseg/semmodel/curve.py already relies on.
    """
    B, N, _ = xyz.shape
    g = xyz - xyz.min(1, keepdim=True).values
    g = (g / g.amax(dim=(1, 2)).clamp_min(EPS).view(B, 1, 1) * (2 ** depth - 1)).floor().long()
    # .contiguous(): torch.linalg.eigh on CUDA returns eigenvectors COLUMN-MAJOR, and that
    # stride survives the slicing and arithmetic downstream; the Hilbert encoder does a
    # dtype-view that requires stride(-1) == 1 and raises otherwise.
    code = encode(g.reshape(-1, 3).contiguous(),
                  torch.arange(B, device=xyz.device).repeat_interleave(N),
                  depth=depth, order=order).reshape(B, N)
    o = torch.argsort(code, dim=1)
    take = torch.linspace(0, N - 1, m, device=xyz.device).long()
    return o[:, take]


# --------------------------------------------------------------------------- #
#  cues and graph                                                              #
# --------------------------------------------------------------------------- #
def hist_embed(fpfh):
    """FPFH histogram [..,33] -> unit vector whose inner product is the Bhattacharyya coefficient.

    open3d's raw FPFH rows are counts (they sum to 600 here); L1-normalise then take the square
    root so that <g_i, g_j> = sum_k sqrt(h_ik h_jk) in [0,1] — the natural similarity between
    two distributions, and it behaves like a cosine everywhere downstream.
    """
    h = fpfh.clamp_min(0)
    h = h / h.sum(-1, keepdim=True).clamp_min(EPS)
    return h.sqrt()


def gather_nb(x, idx):
    """x [B,N,D], idx [B,N,k] -> x[b, idx[b,n,k]] as [B,N,k,D]."""
    B, N, k = idx.shape
    return x.gather(1, idx.reshape(B, N * k, 1).expand(-1, -1, x.shape[-1])).reshape(B, N, k, -1)


def gather_s(v, idx):
    """v [B,N], idx [B,N,k] -> v[b, idx[b,n,k]] as [B,N,k]."""
    B, N, k = idx.shape
    return v.gather(1, idx.reshape(B, N * k)).reshape(B, N, k)


def orient_normals(xyz, nrm, idx, iters=20):
    """Resolve the arbitrary sign of estimated normals so that neighbours agree.

    open3d's `estimate_normals` leaves each normal's sign arbitrary, and the convexity test
    below is meaningless without a consistent orientation.  Maximising sum_ij s_i s_j (n_i.n_j)
    over the kNN graph is an Ising problem; we relax it with sign iterations seeded by the
    outward direction from the shape centroid, which is already right for most of a ShapeNet
    surface.  Returns oriented normals; `agreement` reports the fraction of edges that ended up
    consistent, so a shape where this fails can be spotted rather than silently trusted.
    """
    B, N, k = idx.shape
    c = xyz.mean(1, keepdim=True)
    s = torch.sign((nrm * (xyz - c)).sum(-1))
    s = torch.where(s == 0, torch.ones_like(s), s)
    w = (nrm.unsqueeze(2) * gather_nb(nrm, idx)).sum(-1)              # [B,N,k], sign-free weight
    for _ in range(iters):
        new = torch.sign((w * gather_s(s, idx)).sum(-1))
        new = torch.where(new == 0, s, new)
        if torch.equal(new, s):
            break
        s = new
    m = nrm * s.unsqueeze(-1)
    agree = ((m.unsqueeze(2) * gather_nb(m, idx)).sum(-1) > 0).float().mean()
    return m, float(agree)


def convexity(xyz, m, idx):
    """LCCP's convexity test on every edge: >0 convex, <0 concave.

    c_ij = (m_i - m_j) . (x_i - x_j)/||x_i - x_j||  with CONSISTENTLY ORIENTED normals m.
    On a sphere with outward normals this is +||dx||/R everywhere; on the inside of the same
    sphere it is negative.  Object parts meet at CONCAVE seams (seat/leg, wing/body), which is
    why this, and not feature similarity, is the cue that marks a part boundary.
    """
    dh = F.normalize(xyz.unsqueeze(2) - gather_nb(xyz, idx), dim=-1)
    return ((m.unsqueeze(2) - gather_nb(m, idx)) * dh).sum(-1)


def curve_order(xyz, order, shift, depth=10):
    """Argsort of each shape's points along one space-filling curve."""
    B, N, _ = xyz.shape
    g = xyz - xyz.min(1, keepdim=True).values
    g = (g / g.amax(dim=(1, 2)).clamp_min(EPS).view(B, 1, 1) * (2 ** depth - 1) + shift)
    g = g.floor().long().clamp_(0, 2 ** depth - 1)
    code = encode(g.reshape(-1, 3).contiguous(),
                  torch.arange(B, device=xyz.device).repeat_interleave(N),
                  depth=depth, order=order).reshape(B, N)
    return torch.argsort(code, dim=1)


def curve_knn(xyz, k, depth=10, window=0, orders=ORDERS, shifts=(0.0, 0.5)):
    """Approximate kNN by multi-curve candidate generation -- the object-level counterpart of
    semseg/semmodel/curve.py, batched over shapes.

    Each curve sorts the points and proposes the 2w neighbours inside a window of its order; a
    point adjacent on a space-filling curve is spatially close, and a discontinuity in one
    ordering essentially never lines up with one in another, so the union over eight curves
    ({z, z-trans, hilbert, hilbert-trans} x {origin, half-cell shift}) covers the true
    neighbourhood.  The candidates are then RANKED BY ACTUAL DISTANCE and the top k kept, so the
    curves only have to propose, never to decide -- which is why no vote threshold is needed here
    the way it is at scene scale.

    O(N log N) per curve instead of the O(N^2) distance matrix `knn_idx` builds.
    """
    B, N, _ = xyz.shape
    w = window or max(2, k // 2)
    offs = torch.cat([torch.arange(-w, 0, device=xyz.device),
                      torch.arange(1, w + 1, device=xyz.device)])
    cand = []
    for order in orders:
        for sh in shifts:
            o = curve_order(xyz, order, sh, depth)
            rank = torch.empty_like(o)
            rank.scatter_(1, o, torch.arange(N, device=xyz.device).expand(B, N))
            pos = (rank.unsqueeze(-1) + offs).clamp_(0, N - 1)            # [B,N,2w]
            cand.append(o.gather(1, pos.reshape(B, -1)).reshape(B, N, -1))
    cand = torch.cat(cand, dim=-1)                                        # [B,N,C]
    d2 = (xyz.unsqueeze(2) - gather_nb(xyz, cand)).pow(2).sum(-1)

    # a candidate proposed by several curves appears several times; keep one copy so the top-k is
    # k DISTINCT neighbours, and never the point itself
    cand, srt = torch.sort(cand, dim=-1)
    d2 = d2.gather(-1, srt)
    dup = torch.zeros_like(cand, dtype=torch.bool)
    dup[:, :, 1:] = cand[:, :, 1:] == cand[:, :, :-1]
    d2 = d2.masked_fill(dup | (cand == torch.arange(N, device=xyz.device).view(1, N, 1)),
                        float('inf'))
    sel = d2.topk(k, dim=-1, largest=False)[1]
    return cand.gather(-1, sel)


def knn_idx(xyz, k):
    """Exact kNN by dense cdist; at 2k points this beats any tree.  [B,N,k], self excluded."""
    return torch.cdist(xyz, xyz).topk(k + 1, dim=-1, largest=False)[1][:, :, 1:]


def flat_pairs(idx):
    """[B,N,k] neighbour indices -> symmetric flat (i, j) pairs with batch offsets.

    The batch is flattened to one graph of B*N points (shape b occupies [bN, (b+1)N)); since no
    edge crosses shapes, the flat segment reductions of semseg.semmodel apply unchanged.
    """
    B, N, k = idx.shape
    off = (torch.arange(B, device=idx.device) * N).view(B, 1, 1)
    i = (torch.arange(N, device=idx.device).view(1, N, 1) + off).expand(B, N, k).reshape(-1)
    j = (idx + off).reshape(-1)
    return torch.cat([i, j]), torch.cat([j, i])


def edge_affinity(xyz, nrm, gfe, idx, w_x=1.0, w_n=1.0, w_g=1.0, w_v=0.0, m=None,
                  self_tune=False):
    """Per-edge affinity exp(-sum_cue w_cue d_cue / tau_cue) with tau_cue = the shape's mean d_cue.

        d_x = ||x_i - x_j||^2                 position
        d_n = 1 - |n_i . n_j|                 normal disagreement (sign-free)
        d_g = 1 - <g_i, g_j>                  FPFH histogram disagreement
        d_v = relu(-c_ij)                     CONCAVITY, from the oriented normals `m`

    d_v is the cue that actually marks a part boundary: parts meet at concave seams, so only
    concavity is penalised and a convex edge costs nothing.  `self_tune` replaces the shared
    position scale with Zelnik-Manor local scaling, ||dx||^2/(sigma_i sigma_j) with sigma the
    distance to the k-th neighbour, so a dense region and a sparse one get the same treatment.
    """
    dist2 = (xyz.unsqueeze(2) - gather_nb(xyz, idx)).pow(2).sum(-1)
    if self_tune:
        sig = dist2[:, :, -1].clamp_min(EPS).sqrt()
        dist2 = dist2 / (sig.unsqueeze(2) * gather_s(sig, idx)).clamp_min(EPS)
    d = {
        'x': dist2,
        'n': (1 - (nrm.unsqueeze(2) * gather_nb(nrm, idx)).sum(-1).abs()).clamp_min(0),
        'g': (1 - (gfe.unsqueeze(2) * gather_nb(gfe, idx)).sum(-1)).clamp_min(0),
    }
    if w_v:
        d['v'] = F.relu(-convexity(xyz, nrm if m is None else m, idx))
    e = torch.zeros_like(d['x'])
    for key, w in (('x', w_x), ('n', w_n), ('g', w_g), ('v', w_v)):
        if w and key in d:
            tau = d[key].mean(dim=(1, 2), keepdim=True).clamp_min(EPS)
            e = e - w * d[key] / tau
    return torch.exp(e)


def dense_affinity(w, idx):
    """Scatter the kNN affinities into a symmetric [B,N,N] matrix (max of the two directions)."""
    B, N, _ = idx.shape
    W = w.new_zeros(B, N, N).scatter_(2, idx, w)
    return torch.maximum(W, W.transpose(1, 2))


# --------------------------------------------------------------------------- #
#  spectral embedding + k-means                                                #
# --------------------------------------------------------------------------- #
def spectral_embedding(W, n_ev, eig_dtype=torch.float32):  # noqa: D401
    """Rows of the top-n_ev eigenvectors of D^-1/2 W D^-1/2 (= bottom of the normalised
    Laplacian), L2-normalised.  One batched eigh; a cuSOLVER refusal falls back to the CPU."""
    s = W.sum(-1).clamp_min(EPS).rsqrt()
    A = (s.unsqueeze(2) * W * s.unsqueeze(1)).to(eig_dtype)
    A = 0.5 * (A + A.transpose(1, 2))
    try:
        vec = torch.linalg.eigh(A)[1]
    except Exception:                                        # noqa: BLE001
        vec = torch.linalg.eigh(A.cpu())[1].to(A.device)
    return F.normalize(vec[:, :, -n_ev:].to(W.dtype), dim=-1).contiguous()


def kmeans(X, K, iters=20, init_xyz=None, seed='fps', seed_xyz=None):
    """Batched k-means on [B,N,D].

    Centres are seeded by farthest-point sampling (on `init_xyz` if given, else on X itself,
    from the point farthest from the centroid, so the result is deterministic), or by a
    Hilbert-curve stride when `seed='curve'` -- the same even coverage for a fraction of the
    kernel launches.  The curve is always built on `seed_xyz`, i.e. REAL 3D coordinates: a
    space-filling curve through the columns of a spectral embedding orders nothing meaningful.
    `iters=0` returns the seeds' Voronoi cells.
    """
    base = X if init_xyz is None else init_xyz
    if seed == 'curve':
        sx = seed_xyz if seed_xyz is not None else base
        seeds = curve_sample(sx[:, :, :3], K)
    else:
        seeds = farthest_point_sample(base, K, is_center=True)
    C = index_points(X, seeds)
    lab = None
    for it in range(iters + 1):
        new = torch.cdist(X, C).argmin(-1)
        if it == iters or (lab is not None and torch.equal(new, lab)):
            return new
        lab = new
        oh = F.one_hot(lab, K).to(X.dtype)                                  # [B,N,K]
        cnt = oh.sum(1)                                                     # [B,K]
        Cn = torch.einsum('bnk,bnd->bkd', oh, X) / cnt.clamp_min(1).unsqueeze(-1)
        C = torch.where((cnt > 0).unsqueeze(-1), Cn, C)                     # empty cluster keeps its centre
    return lab


# --------------------------------------------------------------------------- #
#  Sparse spectral: the exact graph, without ever forming an N x N matrix       #
# --------------------------------------------------------------------------- #
def sym_matvec(vals, idx, X):
    """(W + W^T)/2 @ X for W stored as kNN rows: vals [B,N,k], idx [B,N,k], X [B,N,D].

    The kNN graph is directed, and symmetrising it by materialising the transpose would defeat
    the point.  The forward product is a gather, the transpose product is the matching
    scatter-add, and neither ever builds an N x N array.
    """
    B, N, k = idx.shape
    D = X.shape[-1]
    fwd = (gather_nb(X, idx) * vals.unsqueeze(-1)).sum(2)
    bwd = torch.zeros_like(X)
    bwd.scatter_add_(1, idx.reshape(B, N * k, 1).expand(-1, -1, D),
                     (X.unsqueeze(2) * vals.unsqueeze(-1)).reshape(B, N * k, D))
    return 0.5 * (fwd + bwd)


def chol_qr(X, eps=1e-6):
    """Orthonormalise the columns of [B,N,m] via Cholesky-QR.

    Householder QR on a tall matrix is the expensive part of subspace iteration; X^T X is
    [B,m,m], and m is the embedding size (~80), so this is two thin matmuls and a tiny Cholesky.
    """
    G = X.transpose(1, 2) @ X
    G = G + eps * torch.diag_embed(torch.diagonal(G, dim1=1, dim2=2).abs().mean(-1, keepdim=True)
                                   .expand(-1, G.shape[-1]) + eps)
    L = torch.linalg.cholesky(G.double())
    return torch.linalg.solve_triangular(L, X.double().transpose(1, 2), upper=False) \
        .transpose(1, 2).to(X.dtype)


def lobpcg_embedding(W, n_ev, iters=60, eig_dtype=torch.float32):
    """Top-n_ev eigenvectors of D^-1/2 W D^-1/2 by LOBPCG instead of a full eigendecomposition.

    The full `eigh` computes all 2048 eigenpairs to keep 64 of them.  LOBPCG computes only the
    ones asked for, and each of its iterations is a matmul against a thin [N, n_ev] block rather
    than an O(N^3) factorisation.  The matrix is still formed, but forming it is 0.11 ms/shape --
    it was never the cost.
    """
    s = W.sum(-1).clamp_min(EPS).rsqrt()
    A = (s.unsqueeze(2) * W * s.unsqueeze(1)).to(eig_dtype)
    A = 0.5 * (A + A.transpose(1, 2))
    A = A + torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)     # keep it positive definite
    try:
        _, V = torch.lobpcg(A, k=n_ev, largest=True, niter=iters)
    except Exception:                                                  # noqa: BLE001
        V = _eigh(A)[1][:, :, -n_ev:]
    return F.normalize(V.to(W.dtype), dim=-1).contiguous()


def sparse_spectral_embedding(vals, idx, n_ev, iters=30, oversample=16, eig_dtype=torch.float32):
    """Top-n_ev eigenvectors of D^-1/2 W D^-1/2 by subspace iteration on the SPARSE graph.

    The dense path forms an N x N matrix and runs a full eigendecomposition to keep 64 of 2048
    eigenvectors -- 35.1 of 37.6 ms/shape for 3% of the output.  Here the operator is only ever
    applied, never formed: each iteration is a gather and a scatter over the kNN edges, plus a
    Cholesky-QR of an [N, n_ev+oversample] block.  A final Rayleigh-Ritz step rotates the
    converged subspace onto the actual eigenvectors, which matters because k-means is not
    invariant to a rotation of the embedding.

    `oversample` extra columns speed convergence when the leading eigenvalues are clustered,
    which they are for a normalised affinity (everything near 1).
    """
    B, N, k = idx.shape
    dev = vals.device
    ones = torch.ones(B, N, 1, device=dev, dtype=vals.dtype)
    deg = sym_matvec(vals, idx, ones).squeeze(-1).clamp_min(EPS)          # [B,N]
    r = deg.rsqrt()
    nv = vals * r.unsqueeze(-1) * gather_s(r, idx)                        # normalised weights

    m = min(N, n_ev + oversample)
    g = torch.Generator(device='cpu').manual_seed(0)
    X = torch.randn(B, N, m, generator=g).to(dev, vals.dtype)
    X = chol_qr(X)
    for _ in range(iters):
        X = chol_qr(sym_matvec(nv, idx, X) + X)      # (S + I): keeps the spectrum positive
    # Rayleigh-Ritz: diagonalise the operator restricted to the converged subspace
    AX = sym_matvec(nv, idx, X)
    T = X.transpose(1, 2) @ AX
    ev, U = _eigh(0.5 * (T + T.transpose(1, 2)).to(eig_dtype))
    V = X @ U.to(X.dtype)
    return F.normalize(V[:, :, -n_ev:], dim=-1).contiguous()


# --------------------------------------------------------------------------- #
#  Nystrom: the same embedding without the N x N eigendecomposition            #
# --------------------------------------------------------------------------- #
def cue_kernel(A, B, taus, w, self_tune_sig=None):
    """Dense affinity between two point sets, using the same cues as `edge_affinity`.

    A, B are (xyz, oriented normals, FPFH) triples; `taus` are the per-shape cue scales measured
    once on the kNN graph, so the dense kernel and the sparse graph agree on what "far" means.
    Returns [B, |A|, |B|].
    """
    xa, na, ga = A
    xb, nb, gb = B
    d = {}
    d2 = torch.cdist(xa, xb).pow(2)
    if self_tune_sig is not None:
        sa, sb = self_tune_sig
        d2 = d2 / (sa.unsqueeze(2) * sb.unsqueeze(1)).clamp_min(EPS)
    d['x'] = d2
    d['n'] = (1 - torch.einsum('bmd,bnd->bmn', na, nb).abs()).clamp_min(0)
    d['g'] = (1 - torch.einsum('bmd,bnd->bmn', ga, gb)).clamp_min(0)
    if w.get('v'):
        # c_ij = (m_i - m_j).(x_i - x_j)/||x_i - x_j||, expanded so no [M,N,3] tensor is built
        mx_a = (na * xa).sum(-1).unsqueeze(2)                     # m_i . x_i
        mx_b = (nb * xb).sum(-1).unsqueeze(1)                     # m_j . x_j
        c = mx_a - torch.einsum('bmd,bnd->bmn', na, xb)             - torch.einsum('bmd,bnd->bmn', xa, nb) + mx_b
        d['v'] = F.relu(-c / d2.clamp_min(EPS).sqrt())
    e = torch.zeros_like(d['x'])
    for key, wt in w.items():
        if wt and key in d:
            e = e - wt * d[key] / taus[key]
    return torch.exp(e)


def nystrom_embedding(pts, taus, w, n_ev, n_land=256, sig=None, eig_dtype=torch.float32,
                      land='curve', ortho=False):
    """Spectral embedding of every point from an m x m eigendecomposition (Fowlkes et al., 2004).

    The dense path costs one eigh of an N x N matrix per shape, which at N=2048 dominates the
    whole pipeline (38 ms/shape, as much as GeoZe's entire aggregation).  Nystrom samples m
    landmarks, decomposes only the m x m block, and extends the eigenvectors to every point, so
    the cost falls by (N/m)^3 while every point still gets its own embedding row -- the
    partition boundaries stay at full resolution, unlike a coarse-then-propagate scheme.

    `pts` is (xyz, oriented normals, FPFH) for all N points; landmarks are farthest-point
    sampled so they cover the shape.
    """
    xyz, nrm, gfe = pts
    B, N, _ = xyz.shape
    m = min(n_land, N)
    li = curve_sample(xyz, m) if land == 'curve' else farthest_point_sample(xyz, m, is_center=True)
    L = tuple(index_points(t, li) for t in pts)
    sl = None if sig is None else (index_points(sig.unsqueeze(-1), li).squeeze(-1), sig)
    Kaa = cue_kernel(L, L, taus, w, None if sig is None else (sl[0], sl[0]))
    Kab = cue_kernel(L, pts, taus, w, sl)                               # [B, m, N]

    # Row sums of the implied full matrix, with the Nystrom completion C = B^T A^-1 B.
    # Only A^-1 (B 1) is ever needed, so solve for that ONE vector instead of forming the
    # pseudo-inverse: pinv runs an SVD of every [m,m] block and was the single most expensive
    # call in here.
    b1 = Kab.sum(-1).unsqueeze(-1)                                      # [B,m,1]
    eye = torch.eye(m, device=Kaa.device, dtype=eig_dtype).unsqueeze(0)
    x = torch.linalg.solve(Kaa.to(eig_dtype) + 1e-6 * eye, b1.to(eig_dtype)).squeeze(-1)
    d1 = Kaa.sum(-1) + Kab.sum(-1)                                      # [B,m]
    d2 = Kab.sum(1) + torch.einsum('bmn,bm->bn', Kab, x.to(Kab.dtype))
    r1 = d1.clamp_min(EPS).rsqrt()
    r2 = d2.clamp_min(EPS).rsqrt()
    A = Kaa * r1.unsqueeze(2) * r1.unsqueeze(1)
    Bm = Kab * r1.unsqueeze(2) * r2.unsqueeze(1)

    A = 0.5 * (A + A.transpose(1, 2))
    ev, U = _eigh(A.to(eig_dtype))
    Bm = Bm.to(eig_dtype)
    if ortho:
        # Fowlkes' orthogonalised extension: exact eigenvectors of the completed matrix, at the
        # price of a second [m,m] eigendecomposition.
        inv_sqrt = U @ torch.diag_embed(ev.clamp_min(1e-6).rsqrt()) @ U.transpose(1, 2)
        Q = A.to(eig_dtype) + inv_sqrt @ (Bm @ Bm.transpose(1, 2)) @ inv_sqrt
        eq, Uq = _eigh(0.5 * (Q + Q.transpose(1, 2)))
        V = Bm.transpose(1, 2) @ (inv_sqrt @ (Uq @ torch.diag_embed(eq.clamp_min(1e-6).rsqrt())))
    else:
        # Plain Nystrom extension, V = B^T U /  lambda.  One eigendecomposition.  The rows are
        # L2-normalised straight after, and k-means only needs the directions, so the missing
        # orthogonalisation costs little here.
        V = Bm.transpose(1, 2) @ (U * ev.clamp_min(1e-6).reciprocal().unsqueeze(1))
    return F.normalize(V[:, :, -n_ev:].to(xyz.dtype), dim=-1).contiguous()


def _eigh(A):
    try:
        return torch.linalg.eigh(A)
    except Exception:                                        # noqa: BLE001  cuSOLVER refusal
        e, v = torch.linalg.eigh(A.cpu())
        return e.to(A.device), v.to(A.device)


def graph_taus(xyz, nrm, gfe, idx, self_tune=False):
    """The per-shape cue scales, measured on the kNN graph: [B,1,1] each, plus sigma if needed."""
    dist2 = (xyz.unsqueeze(2) - gather_nb(xyz, idx)).pow(2).sum(-1)
    sig = dist2[:, :, -1].clamp_min(EPS).sqrt() if self_tune else None
    if self_tune:
        dist2 = dist2 / (sig.unsqueeze(2) * gather_s(sig, idx)).clamp_min(EPS)
    d = {'x': dist2,
         'n': (1 - (nrm.unsqueeze(2) * gather_nb(nrm, idx)).sum(-1).abs()).clamp_min(0),
         'g': (1 - (gfe.unsqueeze(2) * gather_nb(gfe, idx)).sum(-1)).clamp_min(0),
         'v': F.relu(-convexity(xyz, nrm, idx))}
    return {k: v.mean(dim=(1, 2), keepdim=True).clamp_min(EPS) for k, v in d.items()}, sig


# --------------------------------------------------------------------------- #
#  cleanup                                                                     #
# --------------------------------------------------------------------------- #
def connected_components(lab, idx, max_iter=256):
    """Split each cluster into its connected components on the kNN graph.

    Min-label propagation over both edge directions with pointer jumping; returns the root point
    id of each point's component, [B,N].  Converges in a few iterations for ~30-point clusters.
    """
    B, N, k = idx.shape
    flat = idx.reshape(B, -1)
    same = lab.unsqueeze(2) == lab.gather(1, flat).reshape(B, N, k)
    comp = torch.arange(N, device=lab.device).expand(B, N).clone()
    for _ in range(max_iter):
        pull = comp.gather(1, flat).reshape(B, N, k).masked_fill(~same, N).min(-1)[0]
        push = comp.unsqueeze(2).expand(B, N, k).masked_fill(~same, N).reshape(B, -1)
        new = torch.minimum(comp, pull).scatter_reduce(1, flat, push, 'amin')
        new = new.gather(1, new)                                            # pointer jumping
        if torch.equal(new, comp):
            break
        comp = new
    return comp


def refine_labels(lab, w, idx, K, iters=3):
    """Affinity-weighted majority relabelling on the kNN graph.

    k-means in the spectral embedding places each point independently, so cluster borders come
    out ragged and cut across high-affinity edges.  A few rounds of "take the label your
    strongest neighbours carry" snap the border onto the weak edges of the graph, which is where
    a part seam is.  Cheap: one scatter-add per round, no new graph.
    """
    B, N, k = idx.shape
    for _ in range(iters):
        sc = torch.zeros(B, N, K, device=lab.device, dtype=w.dtype)
        sc.scatter_add_(2, gather_s(lab, idx), w)
        new = sc.argmax(-1)
        if torch.equal(new, lab):
            break
        lab = new
    return lab


def compact(lab):
    """[B,N] arbitrary per-shape ids -> flat [B*N] ids, contiguous and grouped by shape."""
    B, N = lab.shape
    key = torch.arange(B, device=lab.device).unsqueeze(1) * (int(lab.max()) + 1) + lab
    return torch.unique(key, return_inverse=True)[1].reshape(-1)


def absorb_small(seg, idx, min_size, passes=3):
    """Points of regions smaller than `min_size` join the region of their nearest neighbour
    (in kNN order) that belongs to a large region.  seg is flat [B*N]."""
    B, N, k = idx.shape
    nb = (idx + (torch.arange(B, device=idx.device) * N).view(B, 1, 1)).reshape(B * N, k)
    for _ in range(passes):
        cnt = torch.bincount(seg, minlength=int(seg.max()) + 1)
        small = cnt[seg] < min_size
        if not small.any():
            break
        big = ~small[nb]                                                    # [B*N,k]
        first = big.to(torch.uint8).argmax(1)                               # first large neighbour
        cand = seg[nb.gather(1, first.unsqueeze(1)).squeeze(1)]
        seg = torch.where(small & big.any(1), cand, seg)
    return torch.unique(seg, return_inverse=True)[1]


# --------------------------------------------------------------------------- #
#  entry point                                                                 #
# --------------------------------------------------------------------------- #
def superpoints(xyz, nrm, gfe, idx, n_sp=64, method='spectral', w_x=1.0, w_n=1.0, w_g=1.0,
                w_v=0.0, self_tune=False, n_ev=0, orient=True, embed='dense', n_land=256,
                refine=0, split=True, min_size=4, kmeans_iters=20, eig_dtype=torch.float32,
                land='curve', seed='curve', ortho=False, sparse_iters=30):
    """Partition a batch of shapes into superpoints.  Returns flat [B*N] region ids, contiguous
    and grouped by shape, so `seg.max()+1` is the total region count of the batch.

    xyz [B,N,3], nrm [B,N,3] unit, gfe [B,N,33] from hist_embed, idx [B,N,k] from knn_idx.
    `n_ev` is the size of the spectral embedding, independent of the number of clusters `n_sp`
    (0 = use n_sp).  `w_v` weights the concavity cue, which needs `orient=True`.
    """
    B, N, _ = xyz.shape
    if method == 'spectral':
        m = orient_normals(xyz, nrm, idx)[0] if (orient and w_v) else nrm
        if embed == 'lobpcg':
            W = dense_affinity(
                edge_affinity(xyz, nrm, gfe, idx, w_x, w_n, w_g, w_v, m, self_tune), idx)
            emb = lobpcg_embedding(W, n_ev or n_sp, sparse_iters, eig_dtype)
        elif embed == 'sparse':
            w = edge_affinity(xyz, nrm, gfe, idx, w_x, w_n, w_g, w_v, m, self_tune)
            emb = sparse_spectral_embedding(w, idx, n_ev or n_sp, sparse_iters, eig_dtype=eig_dtype)
        elif embed == 'nystrom':
            w = {'x': w_x, 'n': w_n, 'g': w_g, 'v': w_v}
            taus, sig = graph_taus(xyz, m, gfe, idx, self_tune)
            emb = nystrom_embedding((xyz, m, gfe), taus, w, n_ev or n_sp, n_land, sig,
                                    eig_dtype, land, ortho)
        else:
            W = dense_affinity(
                edge_affinity(xyz, nrm, gfe, idx, w_x, w_n, w_g, w_v, m, self_tune), idx)
            emb = spectral_embedding(W, n_ev or n_sp, eig_dtype)
        lab = kmeans(emb, n_sp, kmeans_iters, seed=seed, seed_xyz=xyz)
    elif method == 'kmeans':
        # the same three cues, each scaled to unit mean kNN distance; normals enter through
        # their outer product so the (arbitrary) sign of an estimated normal can not split a cluster
        nn_ = nrm.unsqueeze(-1) * nrm.unsqueeze(-2)
        outer = torch.stack([nn_[..., 0, 0], nn_[..., 1, 1], nn_[..., 2, 2],
                             2 ** 0.5 * nn_[..., 0, 1], 2 ** 0.5 * nn_[..., 0, 2],
                             2 ** 0.5 * nn_[..., 1, 2]], dim=-1)
        cues = []
        for c, w in ((xyz, w_x), (outer, w_n), (gfe, w_g)):
            if w:
                dm = (c.unsqueeze(2) - gather_nb(c, idx)).norm(dim=-1).mean(dim=(1, 2)).clamp_min(EPS)
                cues.append(w * c / dm.view(B, 1, 1))
        lab = kmeans(torch.cat(cues, -1), n_sp, kmeans_iters, init_xyz=xyz, seed=seed,
                     seed_xyz=xyz)
    elif method == 'fps':
        lab = kmeans(xyz, n_sp, 0, seed=seed, seed_xyz=xyz)
    else:
        raise ValueError(f'unknown partition method {method!r}')

    # Boundary refinement is a property of the GRAPH, not of how the labels were first assigned,
    # so it applies to every method.  It is the only way the concavity cue can reach the cheap
    # partitions: k-means works on per-point features and concavity is an edge quantity.
    if refine:
        mm = orient_normals(xyz, nrm, idx)[0] if (orient and w_v) else nrm
        lab = refine_labels(lab, edge_affinity(xyz, nrm, gfe, idx, w_x, w_n, w_g, w_v, mm,
                                               self_tune), idx, n_sp, refine)
    if split:
        lab = lab * N + connected_components(lab, idx)
    seg = compact(lab)
    if min_size > 1:
        seg = absorb_small(seg, idx, min_size)
    return seg
