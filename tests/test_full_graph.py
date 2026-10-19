import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.metrics import f1_score
from scrad.full_graph import best_macro_threshold, load_full_graph

class FullGraphTests(unittest.TestCase):

    def test_threshold_sweep_matches_exhaustive_with_ties(self):
        for seed in range(10):
            rng = np.random.default_rng(seed)
            y = np.r_[np.zeros(27, dtype=int), np.ones(9, dtype=int)]
            scores = rng.integers(0, 8, len(y)) / 7
            threshold = best_macro_threshold(y, scores)
            expected = max((f1_score(y, scores >= t, average='macro') for t in np.r_[np.unique(scores), 2.0]))
            self.assertAlmostEqual(f1_score(y, scores >= threshold, average='macro'), expected)

    def test_full_graph_restores_cross_partition_edges(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            np.savez(path / 'reddit.npz', x=np.arange(12).reshape(4, 3), y=np.array([0, 1, 0, 1]), train=np.array([0]), val=np.array([1]), test=np.array([2, 3]))
            a = sparse.csr_matrix(([1.0] * 6, ([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])), shape=(4, 4))
            sparse.save_npz(path / 'reddit_adj.npz', a)
            part, splits, scaler = load_full_graph(path, 'reddit')
            np.testing.assert_array_equal(part.adj.toarray(), a.toarray())
            np.testing.assert_array_equal(part.edges, [[0, 1], [1, 2], [2, 3]])
            np.testing.assert_array_equal(splits['test'], [2, 3])
            self.assertEqual(part.adj[0, 1], 1)
if __name__ == '__main__':
    unittest.main()
