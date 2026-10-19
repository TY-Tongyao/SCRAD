import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
spec = importlib.util.spec_from_file_location('prediction_auditor', Path(__file__).resolve().parents[1] / 'scripts/audit_complete_predictions.py')
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)

class HeldOutAuditTests(unittest.TestCase):

    def test_reference_vector_refresh_matches_rebuilt_retriever(self):
        from scrad_complete.memory import EntropyRetriever
        rng = np.random.default_rng(71)
        mask = np.ones((2100, 7), bool)
        bank = {'z': rng.normal(size=(2100, 7, 8)).astype(np.float32), 'mask': mask, 'target': np.arange(2100)}
        actual = EntropyRetriever(bank, capacity=8)
        from scrad.retrieval import pooled
        np.testing.assert_array_equal(actual.norm, bank['z'] / np.maximum(np.linalg.norm(bank['z'], axis=-1, keepdims=True), 1e-08))
        np.testing.assert_array_equal(actual.pool, pooled(bank['z'], mask))
        actual.query_batch(bank['z'][:4], mask[:4])
        state = actual.memory.state_dict()
        buffer = actual.norm
        changed = rng.normal(size=(2100, 7, 8)).astype(np.float32)
        expected = EntropyRetriever(dict(bank, z=changed), capacity=8)
        expected.memory.load_state_dict(state)
        actual.update_vectors(changed)
        actual.memory.load_state_dict(state)
        self.assertIs(buffer, actual.norm)
        np.testing.assert_array_equal(actual.norm, expected.norm)
        np.testing.assert_array_equal(actual.pool, expected.pool)
        a, av = actual.query_batch(changed[:12], mask[:12])
        b, bv = expected.query_batch(changed[:12], mask[:12])
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(av, bv)
        np.testing.assert_array_equal(actual.memory.last, expected.memory.last)

    def test_reference_encoding_reuses_ram_and_preserves_read_only_input(self):
        from scrad.model import DualEdgeEncoder
        from scrad_complete.runtime import encode_reference_bank
        torch.manual_seed(71)
        dual = DualEdgeEncoder(4, 3, 2, 0.5)
        context = {'ids': np.arange(3), 'h': torch.randn(3, 4), 'scales': torch.randn(2, 3, 4)}
        bank = {'nodes': np.array([[0, 1, 2], [2, 1, 0]]), 'edge_ids': np.array([[0, 1], [1, 0]]), 'z': np.zeros((2, 5, 4), np.float32)}
        untouched = np.zeros((2, 5, 4), np.float32)
        untouched.flags.writeable = False
        expected = encode_reference_bank(dual, context, dict(bank, z=untouched), 'cpu')
        actual = encode_reference_bank(dual, context, bank, 'cpu')
        self.assertIs(actual, bank['z'])
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(untouched.any())

    def fixture(self, root):
        run, evaluation, data = (root / key for key in ['run', 'evaluation', 'data'])
        for path in [run / 'bank', evaluation, data]:
            path.mkdir(parents=True)
        (run / 'prepared.json').write_text(json.dumps({'dataset': 'questions', 'seed': 42, 'label_fraction': 0, 'valid_held_out_evaluation': True}))
        (evaluation / 'checkpoint_used.pt').write_bytes(b'fixture checkpoint')
        (evaluation / 'config.json').write_text(json.dumps({'decision_mode': 'joint', 'candidates': 0, 'limit_nodes': 0, 'checkpoint_sha256': hashlib.sha256(b'fixture checkpoint').hexdigest()}))
        (evaluation / 'report.json').write_text(json.dumps({'complete': True, 'complete_original_test': True, 'decision_mode': 'joint', 'exact_full_corpus_retrieval': True, 'threshold': 0.5, 'encoder_score_used_as_prediction': False, 'valid_held_out_evaluation': True, 'nodes': 2, 'metrics': {'auroc': 1.0, 'average_precision': 1.0, 'macro_f1': 1.0, 'nodes': 2, 'positives': 1, 'predicted_positives': 1}}))
        rows = [{'id': i, 'sequence_labels': [label] * 9, 'score': float(label), 'predicted_label': label, 'llm_probabilities_diagnostic_only': [0.9 if label else 0.1] * 9} for i, label in [(3, 0), (4, 1)]]
        (evaluation / 'predictions.jsonl').write_text(''.join((json.dumps(row) + '\n' for row in rows)))
        torch.save({'completed_nodes': 2}, evaluation / 'memory.pt')
        np.save(run / 'test_ids.npy', [3, 4])
        np.save(run / 'bank/target.npy', [0, 1])
        np.save(run / 'bank/y.npy', [0, 1])
        np.save(run / 'bank/nodes.npy', [[0, 1, -1], [1, 0, -1]])
        np.savez(run / 'evaluation.npz', node_ids=[3, 4], y=[0, 1], val_ids=[2], val_y=[0])
        np.savez(run / 'supervision.npz', ids=[0, 1], y=[0, 1])
        np.savez(data / 'questions.npz', y=[0, 1, 0, 0, 1], train=[0, 1], val=[2], test=[3, 4])
        np.savez(data / 'injection.npz', structural_nodes=np.array([4]), contextual_nodes=np.array([], dtype=np.int64))
        (data / 'protocol.json').write_text(json.dumps({'test_metrics_used_for_injection_selection': False}))
        return (run, evaluation, data)

    def test_clean_fixture_and_reject_test_center_in_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            run, evaluation, data = self.fixture(Path(directory))
            self.assertTrue(auditor.audit(run, evaluation, data)['valid_held_out_evaluation'])
            np.save(run / 'bank/target.npy', [0, 3])
            with self.assertRaisesRegex(AssertionError, 'nontraining centers'):
                auditor.audit(run, evaluation, data)
            np.save(run / 'bank/target.npy', [0, 1])
            np.save(run / 'bank/nodes.npy', [[0, 3, -1], [1, 0, -1]])
            with self.assertRaisesRegex(AssertionError, 'nontraining nodes'):
                auditor.audit(run, evaluation, data)

    def test_reject_test_supervision_and_test_selected_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            run, evaluation, data = self.fixture(Path(directory))
            np.savez(run / 'supervision.npz', ids=[0, 3], y=[0, 0])
            with self.assertRaisesRegex(AssertionError, 'test labels'):
                auditor.audit(run, evaluation, data)
            np.savez(run / 'supervision.npz', ids=[0, 1], y=[0, 1])
            (data / 'protocol.json').write_text(json.dumps({'test_metrics_used_for_injection_selection': True}))
            with self.assertRaises(AssertionError):
                auditor.audit(run, evaluation, data)

    def test_reject_changed_bank_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            run, evaluation, data = self.fixture(Path(directory))
            np.save(run / 'bank/y.npy', [1, 0])
            with self.assertRaises(AssertionError):
                auditor.audit(run, evaluation, data)

    def test_reject_different_source_test_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            run, evaluation, data = self.fixture(Path(directory))
            np.savez(data / 'questions.npz', y=[0, 1, 0, 0, 1], train=[0, 1], val=[3], test=[2, 4])
            with self.assertRaises(AssertionError):
                auditor.audit(run, evaluation, data)
if __name__ == '__main__':
    unittest.main()
