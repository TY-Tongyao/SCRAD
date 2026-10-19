import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
from scrad.data import load_partitions, require_binary_anomaly_labels
from scrad.full_graph import load_full_graph

class DatasetSourceTests(unittest.TestCase):

    def test_binary_anomaly_labels_remain_supported(self):
        require_binary_anomaly_labels(np.array([0, 1, 0]), 'questions')

    def test_community_labels_rejected_before_graph_or_features_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory) / 'reddit.npz', y=np.arange(41))
            with patch('scipy.sparse.load_npz', side_effect=AssertionError('graph read')):
                for loader in (load_partitions, load_full_graph):
                    with self.subTest(loader=loader.__name__):
                        with self.assertRaisesRegex(ValueError, 'community classes'):
                            loader(directory, 'reddit')

    def test_edge_normalization_removes_loops_and_preserves_both_directions(self):
        scripts = Path(__file__).resolve().parents[1] / 'scripts'
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location('prepare_dataset_sources', scripts / 'prepare_data.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(scripts))
        raw = sparse.coo_matrix((np.array([3.0, 2.0, 1.0, 4.0]), (np.array([0, 0, 1, 2]), np.array([1, 1, 1, 1]))), shape=(3, 3))
        graph, stats = module.normalize_adjacency(raw)
        np.testing.assert_array_equal(graph.toarray(), [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        self.assertEqual(stats['raw_edge_records'], 4)
        self.assertEqual(stats['raw_unique_directed_edges'], 3)
        self.assertEqual(stats['raw_self_loops'], 1)
        self.assertEqual(stats['undirected_edges_without_loops'], 2)
if __name__ == '__main__':
    unittest.main()
