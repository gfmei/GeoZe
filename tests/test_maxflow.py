import itertools
import unittest

import torch

from common.maxflow import max_flow


class MaxFlowTests(unittest.TestCase):
    def check_graph(self, edges, capacity, n, source=0, sink=None):
        sink = n - 1 if sink is None else sink
        result = max_flow(edges, capacity, source, sink, n, max_iterations=10000)
        # Independent oracle: enumerate all terminal-separating cuts.
        others = [v for v in range(n) if v not in (source, sink)]
        minimum = float("inf")
        for bits in itertools.product((False, True), repeat=len(others)):
            side = torch.zeros(n, dtype=torch.bool, device=capacity.device)
            side[source] = True
            for v, bit in zip(others, bits):
                side[v] = bit
            cut = capacity[side[edges[0]] & ~side[edges[1]]].sum().item()
            minimum = min(minimum, cut)
        self.assertAlmostEqual(result.value.item(), minimum, places=5)
        self.assertTrue(bool((result.flow >= -1e-6).all()))
        self.assertTrue(bool((result.flow <= capacity + 1e-6).all()))
        balance = capacity.new_zeros(n)
        balance.index_add_(0, edges[0], -result.flow)
        balance.index_add_(0, edges[1], result.flow)
        expected = torch.zeros_like(balance)
        expected[source] = -result.value
        expected[sink] = result.value
        torch.testing.assert_close(balance, expected, atol=1e-6, rtol=1e-6)
        self.assertTrue(result.source_side[source].item())
        self.assertFalse(result.source_side[sink].item())
        cut = capacity[result.source_side[edges[0]] & ~result.source_side[edges[1]]].sum()
        torch.testing.assert_close(cut, result.value, atol=1e-6, rtol=1e-6)
        return result

    def test_classic(self):
        edges = torch.tensor([[0, 0, 1, 2, 1, 3, 2, 4, 4, 3],
                              [1, 2, 2, 1, 3, 2, 4, 3, 5, 5]])
        cap = torch.tensor([16, 13, 10, 4, 12, 9, 14, 7, 4, 20], dtype=torch.float64)
        self.assertEqual(self.check_graph(edges, cap, 6).value.item(), 23)

    def test_random_and_empty(self):
        generator = torch.Generator().manual_seed(42)
        for dtype in (torch.float32, torch.float64):
            for n in range(2, 8):
                for trial in range(15):
                    m = 0 if trial == 0 else n * 3
                    edges = torch.randint(n, (2, m), generator=generator)
                    cap = torch.randint(0, 20, (m,), generator=generator).to(dtype) / 8
                    original = cap.clone()
                    self.check_graph(edges, cap, n, source=n - 1, sink=0)
                    torch.testing.assert_close(cap, original)

    def test_validation_and_round_limit(self):
        edges = torch.tensor([[0, 1], [1, 2]])
        with self.assertRaises(ValueError):
            max_flow(edges, torch.tensor([-1., 1.]), 0, 2, 3)
        with self.assertRaises(ValueError):
            max_flow(edges, torch.ones(2), 0, 0, 3)
        with self.assertRaises(RuntimeError):
            max_flow(edges, torch.ones(2), 0, 2, 3, max_iterations=1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda(self):
        generator = torch.Generator().manual_seed(7)
        for _ in range(10):
            edges = torch.randint(6, (2, 24), generator=generator).cuda()
            capacity = torch.rand(24, generator=generator, dtype=torch.float64).cuda()
            self.check_graph(edges, capacity, 6)


if __name__ == "__main__":
    unittest.main()
