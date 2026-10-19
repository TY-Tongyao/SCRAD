import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
SPEC = importlib.util.spec_from_file_location('prediction_audit', Path(__file__).resolve().parents[1] / 'scripts/audit_complete_predictions.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

class PredictionAuditTests(unittest.TestCase):

    def fixture(self, root, mode):
        run = root / 'run'
        evaluation = root / 'evaluation'
        data = root / 'data'
        for directory in [run / 'bank', evaluation, data]:
            directory.mkdir(parents=True)
        y = np.asarray([0, 1] * 5)
        train, val, test = (np.arange(4), np.arange(4, 6), np.arange(6, 10))
        scores = np.asarray([0.0, 1.0, 1.0, 1.0])
        (run / 'prepared.json').write_text(json.dumps({'dataset': 'questions', 'seed': 42, 'label_fraction': 0, 'valid_held_out_evaluation': True}))
        np.save(run / 'test_ids.npy', test)
        np.savez(run / 'evaluation.npz', node_ids=test, y=y[test], val_ids=val, val_y=y[val])
        np.savez(run / 'supervision.npz', ids=train, y=y[train])
        np.save(run / 'bank/target.npy', train[:2])
        np.save(run / 'bank/y.npy', y[:2])
        np.save(run / 'bank/nodes.npy', np.asarray([[0, 2], [1, 3]]))
        np.savez(data / 'questions.npz', y=y, train=train, val=val, test=test)
        np.savez(data / 'injection.npz', structural_nodes=np.asarray([1, 7]), contextual_nodes=np.asarray([3, 9]))
        (data / 'protocol.json').write_text(json.dumps({'test_metrics_used_for_injection_selection': False}))
        checkpoint = b'Unit-test fixture, not a trained model'
        (evaluation / 'checkpoint_used.pt').write_bytes(checkpoint)
        (evaluation / 'config.json').write_text(json.dumps({'decision_mode': mode, 'candidates': 0, 'limit_nodes': 0, 'checkpoint_sha256': hashlib.sha256(checkpoint).hexdigest()}))
        measured = AUDIT.measure(y[test], scores)
        (evaluation / 'report.json').write_text(json.dumps({'complete': True, 'complete_original_test': True, 'decision_mode': mode, 'exact_full_corpus_retrieval': True, 'threshold': 0.5, 'encoder_score_used_as_prediction': False, 'valid_held_out_evaluation': True, 'nodes': len(test), 'metrics': measured}))
        rows = [{'id': int(i), 'sequence_labels': [int(score)] * 9, 'score': float(score), 'predicted_label': int(score), 'llm_probabilities_diagnostic_only': [0.9 if score else 0.1] * 9} for i, score in zip(test, scores)]
        (evaluation / 'predictions.jsonl').write_text(''.join((json.dumps(row) + '\n' for row in rows)))
        torch.save({'completed_nodes': len(test)}, evaluation / 'memory.pt')
        return (run, evaluation, data)

    def test_both_modes_keep_identical_score_and_label_boundaries(self):
        for mode in ['joint', 'independent']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                result = AUDIT.audit(*self.fixture(Path(directory), mode))
                self.assertTrue(result['verified_complete'])
                self.assertEqual(result['decision_mode'], mode)
                self.assertEqual(result['metrics']['confusion_matrix_true_rows_predicted_columns'], [[1, 1], [0, 2]])
                self.assertEqual(result['sequence_decisions'], 36)

    def test_mode_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run, evaluation, data = self.fixture(Path(directory), 'joint')
            report = json.loads((evaluation / 'report.json').read_text())
            report['decision_mode'] = 'independent'
            (evaluation / 'report.json').write_text(json.dumps(report))
            with self.assertRaises(AssertionError):
                AUDIT.audit(run, evaluation, data)
if __name__ == '__main__':
    unittest.main()
