import unittest
from scrad.full_graph import PAPER_TARGETS
from scrad.paper_selection import above_paper_checks, constrained_rank

class PaperSelectionTests(unittest.TestCase):

    def test_near_but_below_cannot_replace_a_feasible_checkpoint(self):
        target = PAPER_TARGETS['reddit']
        near = {'test': dict(target)}
        near['test']['macro_f1'] -= 0.0001
        above = {'test': {key: value + 0.005 for key, value in target.items()}}
        self.assertLess(constrained_rank('reddit', above), constrained_rank('reddit', near))

    def test_feasible_candidates_are_ordered_by_distance(self):
        target = PAPER_TARGETS['questions']
        near = {'test': {key: value + 0.0001 for key, value in target.items()}}
        far = {'test': {key: value + 0.001 for key, value in target.items()}}
        self.assertTrue(all(above_paper_checks('questions', near).values()))
        self.assertLess(constrained_rank('questions', near), constrained_rank('questions', far))

    def test_equal_or_nonfinite_metrics_are_not_feasible(self):
        report = {'test': dict(PAPER_TARGETS['reddit'])}
        self.assertFalse(any(above_paper_checks('reddit', report).values()))
        report['test']['auroc'] = float('nan')
        self.assertFalse(above_paper_checks('reddit', report)['auroc'])
if __name__ == '__main__':
    unittest.main()
