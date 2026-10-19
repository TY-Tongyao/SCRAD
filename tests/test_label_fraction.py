import unittest
import numpy as np
from scrad.label_fraction import supervised_subset, evaluate_fraction_scores

class LabelFractionTests(unittest.TestCase):

    def setUp(self):
        self.y = np.tile([0, 0, 0, 1], 30)
        self.splits = dict(train=np.arange(40), val=np.arange(40, 80), test=np.arange(80, 120))

    def test_zero_full_and_nested_subsets(self):
        previous = set()
        for fraction in (0, 0.1, 0.25, 0.5, 0.75, 1):
            ids, counts = supervised_subset(self.y, self.splits, fraction)
            self.assertTrue(set(self.splits['train']) <= set(ids))
            self.assertTrue(previous <= set(ids))
            self.assertEqual(len(ids), len(set(ids)))
            for name in ('val', 'test'):
                self.assertEqual(len(np.intersect1d(ids, self.splits[name])), counts[name]['supervised_nodes'])
            previous = set(ids)
            if fraction == 0:
                np.testing.assert_array_equal(ids, self.splits['train'])
            if fraction == 1:
                np.testing.assert_array_equal(ids, np.arange(len(self.y)))

    def test_calibration_does_not_read_withheld_labels(self):
        ids, _ = supervised_subset(self.y, self.splits, 0.25)
        scores = np.random.default_rng(3).random(len(self.y))
        first = evaluate_fraction_scores(self.y, scores, self.splits, ids)
        changed = self.y.copy()
        withheld = np.setdiff1d(np.arange(len(self.y)), ids)
        changed[withheld] = 1 - changed[withheld]
        second = evaluate_fraction_scores(changed, scores, self.splits, ids)
        self.assertEqual(first['threshold'], second['threshold'])
        self.assertEqual(first['all_supervised_nodes'], second['all_supervised_nodes'])
        self.assertFalse(first['valid_held_out_evaluation'])
if __name__ == '__main__':
    unittest.main()
