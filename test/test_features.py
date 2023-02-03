import unittest
import numpy as np

from src.features.compute_semantic_stats import compute_running_stats


class TestSemanticFeatures(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 10**7
        self.x = [np.random.rand(self.n), np.random.rand(self.n)]

    def test_running_stats(self):
        stats = compute_running_stats(self.x)
        x = np.array(self.x).flatten()
        np.testing.assert_allclose(stats.get('mean'), np.mean(x), rtol=1e-6)
        np.testing.assert_allclose(stats.get('std'), np.std(x), rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
