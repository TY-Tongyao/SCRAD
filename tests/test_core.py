import unittest
import numpy as np
from scipy import sparse
from scrad.data import Partition, sample_sequences, line_graph_scales
from scrad.retrieval import entropy_similarity, Retriever, shared_path, normalized_matching_scores
from scrad.infer import parse_predictions

class CoreTests(unittest.TestCase):

    def part(self):
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
        u, v = edges.T
        a = sparse.csr_matrix((np.ones(8), (np.r_[u, v], np.r_[v, u])), shape=(6, 6))
        ids = np.arange(1, 5)
        lookup = sparse.csr_matrix((np.r_[ids, ids], (np.r_[u, v], np.r_[v, u])), shape=a.shape)
        return Partition(np.arange(6), np.ones((6, 2)), np.zeros(6), a, edges, lookup)

    def test_dfs_no_false_edges_and_isolate(self):
        p = self.part()
        centers = np.tile(np.arange(6), 10)
        s, e = sample_sequences(p, centers, np.random.default_rng(42))
        np.testing.assert_array_equal(s[:, 3], centers)
        for row, erow in zip(s, e):
            nodes = row[row >= 0]
            self.assertEqual(len(nodes), len(set(nodes.tolist())))
            for a, b in zip(nodes[:-1], nodes[1:]):
                self.assertEqual(p.adj[a, b], 1)
        self.assertEqual(int((s[centers == 5] >= 0).sum()), 10)
        self.assertTrue((e[centers == 5] == -1).all())

    def test_implicit_line_graph_matches_explicit(self):
        p = self.part()
        h = np.arange(12, dtype=np.float32).reshape(6, 2)
        e = p.edges
        init = (h[e[:, 0]] + h[e[:, 1]]) / 2
        b = np.zeros((6, 4))
        for i, (u, v) in enumerate(e):
            b[u, i] = b[v, i] = 1
        a = b.T @ b - np.eye(4)
        d = a.sum(1) ** (-0.5)
        normalized = d[:, None] * a * d[None, :]
        result = line_graph_scales(p, h, depth=1, restart=0.1)[:, 0]
        np.testing.assert_allclose(result, 0.9 * normalized @ init + 0.1 * init, rtol=1e-05)

    def test_entropy_penalizes_uneven_matches(self):
        a = np.tile([1.0, 0.0], (5, 1))
        uniform = np.tile([0.6, 0.8], (5, 1))
        uneven = np.array([[1.0, 0.0]] * 4 + [[-1.0, 0.0]])
        self.assertGreater(entropy_similarity(a, uniform), entropy_similarity(a, uneven))
        self.assertEqual(entropy_similarity(a, -a), 0)
        self.assertTrue(np.isfinite(entropy_similarity(a[:1], uniform)))

    def test_matching_permutation_invariant(self):
        a = np.eye(3)
        self.assertAlmostEqual(entropy_similarity(a, a), entropy_similarity(a, a[::-1]))

    def test_vectorized_matching_agrees_with_scalar(self):
        rng = np.random.default_rng(9)
        for n, m in [(1, 13), (13, 1), (3, 7), (13, 13)]:
            a = rng.normal(size=(n, 8))
            b = rng.normal(size=(5, m, 8))
            expected = [entropy_similarity(a, item) for item in b]
            a /= np.linalg.norm(a, axis=-1, keepdims=True)
            b /= np.linalg.norm(b, axis=-1, keepdims=True)
            np.testing.assert_allclose(normalized_matching_scores(a, b), expected, rtol=1e-06)

    def test_no_paths_between_disjoint_partitions(self):
        self.assertFalse(shared_path(np.array([1, 2, -1]), np.array([3, 4, -1]))['reachable'])
        self.assertEqual(shared_path(np.array([1, 2]), np.array([2, 3]))['shared_nodes'], [2])

    def test_evaluation_retrieval_does_not_change_usefulness(self):
        z = np.array([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]])
        bank = {'z': z, 'mask': np.ones((2, 2), bool), 'utility': np.array([1.0, 0.0])}
        r = Retriever(bank, candidates=0, topk=1)
        r.query_batch(z, np.ones((2, 2), bool))
        np.testing.assert_array_equal(r.utility, bank['utility'])
        self.assertEqual(r.frequency.sum(), 2)

    def test_api_rejects_invalid_missing_duplicate_labels(self):
        good = '{"predictions":[{"id":2,"labels":[0,1]}]}'
        self.assertEqual(parse_predictions(good, [2], 2), {2: [0, 1]})
        for bad in [good.replace('[0,1]', '[0]'), good.replace('[0,1]', '[0,2]'), good.replace('[0,1]', '[false,1]'), good.replace('"id":2', '"id":3')]:
            with self.assertRaises(ValueError):
                parse_predictions(bad, [2], 2)
if __name__ == '__main__':
    unittest.main()
