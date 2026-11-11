"""Two focused CPU checks for circular indexing and the reference residual boundary."""
import unittest
import torch
from runtime import circular_indices
from reference_binding import align_reference_geometry


class ReferenceTests(unittest.TestCase):
    def test_all_starts_and_boundary_order(self):
        self.assertEqual(circular_indices(72), list(range(72, 80)) + list(range(9)))
        self.assertEqual(circular_indices(78), [78, 79] + list(range(15)))
        for start in range(80):
            ids = circular_indices(start)
            self.assertEqual(len(set(ids)), 17)
            self.assertTrue(all(0 <= i < 80 for i in ids))
            self.assertTrue(all((b - a) % 80 == 1 for a, b in zip(ids, ids[1:])))

    def test_reference_prefix_zero_after_biased_projection(self):
        tokens, gate = torch.randn(1, 20, 3), torch.rand(1, 20, 1)
        projection = torch.nn.Linear(3, 7)
        with torch.no_grad():
            projection.bias.fill_(2)
        padded, padded_gate = align_reference_geometry(tokens, gate, 5, 1)
        self.assertEqual(tuple(padded.shape), (1, 24, 3))
        delta = projection(padded) * padded_gate
        self.assertTrue(torch.equal(delta[:, :4], torch.zeros_like(delta[:, :4])))
        self.assertTrue(torch.equal(padded[:, 4:], tokens))
        self.assertTrue(torch.equal(padded_gate[:, 4:], gate))
        torch.testing.assert_close(delta[:, 4:], projection(tokens) * gate)
        unchanged = align_reference_geometry(tokens, gate, 5, 0)
        self.assertIs(unchanged[0], tokens)
        self.assertIs(unchanged[1], gate)


if __name__ == "__main__":
    unittest.main()
