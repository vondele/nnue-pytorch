import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.dataset import MixedSparseBatchProvider


class FakeBatchIterator:
    """Simple infinite iterator that returns a constant label."""

    def __init__(self, label):
        self.label = label

    def __iter__(self):
        return self

    def __next__(self):
        return self.label


class TestMixedSparseBatchProvider(unittest.TestCase):
    def test_distribution_matches_fraction(self):
        """Draw many batches and check the CDB ratio is close to cdb_fraction."""
        cdb_fraction = 0.3
        provider = MixedSparseBatchProvider(
            FakeBatchIterator("binpack"),
            FakeBatchIterator("cdb"),
            cdb_fraction,
            seed=12345,
        )

        counts = {"binpack": 0, "cdb": 0}
        n = 2000
        for _ in range(n):
            counts[next(provider)] += 1

        self.assertEqual(counts["binpack"] + counts["cdb"], n)
        observed = counts["cdb"] / n
        self.assertAlmostEqual(observed, cdb_fraction, delta=0.05)

    def test_zero_fraction_uses_only_binpack(self):
        provider = MixedSparseBatchProvider(
            FakeBatchIterator("binpack"),
            FakeBatchIterator("cdb"),
            0.0,
            seed=0,
        )
        for _ in range(100):
            self.assertEqual(next(provider), "binpack")

    def test_one_fraction_uses_only_cdb(self):
        provider = MixedSparseBatchProvider(
            FakeBatchIterator("binpack"),
            FakeBatchIterator("cdb"),
            1.0,
            seed=0,
        )
        for _ in range(100):
            self.assertEqual(next(provider), "cdb")

    def test_per_rank_seed_changes_choices(self):
        """Different ranks should make different source choices."""
        provider0 = MixedSparseBatchProvider(
            FakeBatchIterator("binpack"),
            FakeBatchIterator("cdb"),
            0.5,
            seed=42,
        )
        provider1 = MixedSparseBatchProvider(
            FakeBatchIterator("binpack"),
            FakeBatchIterator("cdb"),
            0.5,
            seed=43,
        )

        choices0 = [next(provider0) for _ in range(20)]
        choices1 = [next(provider1) for _ in range(20)]
        self.assertNotEqual(choices0, choices1)


if __name__ == "__main__":
    unittest.main()
