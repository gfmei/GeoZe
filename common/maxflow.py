"""Sparse single-source/sink maximum flow using PyTorch push–relabel.

Pass CUDA tensors to execute tensor operations on the GPU. Memory is O(V + E).
This is a pure-PyTorch baseline: each round scans the residual edges and the
Python convergence check synchronizes the device. It is not a custom CUDA
solver and does not guarantee a speedup over optimized CPU solvers.

Example::

    edges = torch.tensor([[0, 0, 1, 2, 1], [1, 2, 3, 3, 2]], device="cuda")
    capacities = torch.tensor([3., 2., 2., 3., 1.], device="cuda")
    result = max_flow(edges, capacities, source=0, sink=3, num_nodes=4)
    print(result.value.item())  # 5.0
"""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class MaxFlowResult:
    value: torch.Tensor
    flow: torch.Tensor
    source_side: torch.Tensor
    iterations: int


@torch.no_grad()
def max_flow(
    edge_index: torch.Tensor,
    capacity: torch.Tensor,
    source: int,
    sink: int,
    num_nodes: int,
    *,
    atol: float = 0.0,
    max_iterations: int | None = None,
) -> MaxFlowResult:
    """Compute a directed maximum flow and its residual minimum cut.

    Args:
        edge_index: Long tensor of shape (2, E), containing tails and heads.
        capacity: Finite nonnegative float32/float64 tensor of shape (E,), on
            the same device as edge_index. Inputs are never modified.
        source, sink: Distinct terminal node indices.
        num_nodes: Number of vertices, including isolated vertices.
        atol: Residual/excess threshold; zero by default. A positive threshold
            yields an approximate solution and can discard small capacities.
        max_iterations: Optional round limit; raises on non-convergence.

    Returns:
        Scalar flow value, flow in input edge order, boolean source-side cut
        mask, and number of push/relabel rounds, all tensors on the input device.

    Parallel edges, antiparallel edges, and self-loops are supported. Self-loops
    carry zero flow. For an undirected capacity, supply both directed edges.
    This discrete solver is not differentiable. Floating-point roundoff and
    CUDA scatter-add ordering can affect results; prefer float64 when needed.
    """
    if not isinstance(num_nodes, int) or num_nodes < 2:
        raise ValueError("num_nodes must be an integer >= 2")
    if not isinstance(source, int) or not isinstance(sink, int):
        raise ValueError("source and sink must be integers")
    if not (0 <= source < num_nodes and 0 <= sink < num_nodes) or source == sink:
        raise ValueError("source and sink must be distinct valid node indices")
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be a long tensor of shape (2, E)")
    if capacity.ndim != 1 or capacity.numel() != edge_index.shape[1]:
        raise ValueError("capacity must have shape (E,)")
    if capacity.dtype not in (torch.float32, torch.float64):
        raise ValueError("capacity must use float32 or float64")
    if edge_index.device != capacity.device:
        raise ValueError("edge_index and capacity must be on the same device")
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    if max_iterations is not None and (
        not isinstance(max_iterations, int) or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer or None")
    if bool(((edge_index < 0) | (edge_index >= num_nodes)).any()):
        raise ValueError("edge_index contains an invalid node index")
    if bool((~torch.isfinite(capacity) | (capacity < 0)).any()):
        raise ValueError("capacities must be finite and nonnegative")
    if not bool(torch.isfinite(capacity.sum())):
        raise ValueError("total capacity overflows the capacity dtype")

    device = capacity.device
    m = capacity.numel()
    tail = torch.cat((edge_index[0], edge_index[1]))
    head = torch.cat((edge_index[1], edge_index[0]))
    ids = torch.arange(2 * m, device=device)
    reverse = torch.cat((ids[m:], ids[:m]))
    residual = torch.cat((capacity.clone(), torch.zeros_like(capacity)))
    height = torch.zeros(num_nodes, dtype=torch.long, device=device)
    height[source] = num_nodes
    excess = capacity.new_zeros(num_nodes)

    # Saturate outgoing source edges to construct the initial preflow.
    initial = torch.where(
        (tail == source) & (head != source), residual, 0.0
    )
    residual -= initial
    residual.index_add_(0, reverse, initial)
    excess.index_add_(0, head, initial)
    excess.index_add_(0, tail, -initial)

    iterations = 0
    sentinel = 2 * m
    while True:
        active = excess > atol
        active[source] = False
        active[sink] = False
        if not bool(active.any()):
            break
        if max_iterations is not None and iterations >= max_iterations:
            raise RuntimeError(f"max_flow did not converge in {max_iterations} rounds")

        usable = (residual > atol) & (tail != head)
        admissible = usable & active[tail] & (height[tail] == height[head] + 1)
        # One outgoing push per active vertex avoids overspending its excess.
        # Opposite residual arcs cannot both be admissible in the same round.
        chosen = torch.full((num_nodes,), sentinel, dtype=torch.long, device=device)
        chosen.scatter_reduce_(
            0, tail, torch.where(admissible, ids, sentinel), reduce="amin"
        )
        selected = ids[admissible & (ids == chosen[tail])]
        amount = torch.minimum(excess[tail[selected]], residual[selected])

        # Compute relabels from the snapshot before applying any pushes.
        nearest = torch.full((num_nodes,), 2 * num_nodes, dtype=torch.long, device=device)
        nearest.scatter_reduce_(
            0, tail, torch.where(usable, height[head], 2 * num_nodes), reduce="amin"
        )
        relabel = active & (chosen == sentinel)
        height = torch.where(relabel, nearest + 1, height)

        residual.index_add_(0, selected, -amount)
        residual.index_add_(0, reverse[selected], amount)
        excess.index_add_(0, tail[selected], -amount)
        excess.index_add_(0, head[selected], amount)
        iterations += 1

    # Reachability from source in the final residual graph identifies the cut.
    # Held as uint8 rather than bool: scatter_reduce_ has no CUDA kernel for Bool
    # ("cuda_scatter_gather_base_kernel_func not implemented for 'Bool'"), so a bool
    # accumulator works on CPU and raises on the device this is meant to run on.
    reach = torch.zeros(num_nodes, dtype=torch.uint8, device=device)
    reach[source] = 1
    usable = residual > atol
    while True:
        reached = reach.clone()
        reached.scatter_reduce_(
            0, head, (reach[tail].bool() & usable).to(torch.uint8), reduce="amax"
        )
        if torch.equal(reached, reach):
            break
        reach = reached

    return MaxFlowResult(excess[sink], residual[m:], reach.bool(), iterations)
