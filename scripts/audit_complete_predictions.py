import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
TARGETS = {'reddit': {'auroc': 0.9772, 'average_precision': 0.8945, 'macro_f1': 0.9302}, 'questions': {'auroc': 0.9598, 'average_precision': 0.829, 'macro_f1': 0.911}}

def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def measure(labels, scores):
    binary = scores >= 0.5
    both_classes = len(np.unique(labels)) == 2
    return {'nodes': len(labels), 'positives': int(labels.sum()), 'predicted_positives': int(binary.sum()), 'auroc': float(roc_auc_score(labels, scores)) if both_classes else None, 'average_precision': float(average_precision_score(labels, scores)) if both_classes else None, 'macro_f1': float(f1_score(labels, binary, average='macro', labels=[0, 1], zero_division=0)), 'confusion_matrix_true_rows_predicted_columns': confusion_matrix(labels, binary, labels=[0, 1]).tolist()}

def audit(run, evaluation, data=None):
    run, evaluation = (Path(run), Path(evaluation))
    prepared = json.loads((run / 'prepared.json').read_text())
    config = json.loads((evaluation / 'config.json').read_text())
    report = json.loads((evaluation / 'report.json').read_text())
    assert report['complete'] and report['complete_original_test'], 'Full evaluation is not finished'
    assert report['decision_mode'] == config['decision_mode']
    assert config['decision_mode'] in ['joint', 'independent']
    assert report['exact_full_corpus_retrieval'] and config['candidates'] == 0
    assert config['limit_nodes'] == 0 and report['threshold'] == 0.5
    assert not report['encoder_score_used_as_prediction']
    held_out = bool(report['valid_held_out_evaluation'])
    if held_out:
        assert prepared.get('valid_held_out_evaluation') and prepared['label_fraction'] == 0
    checkpoint_sha = digest(evaluation / 'checkpoint_used.pt')
    assert checkpoint_sha == config['checkpoint_sha256'], 'Checkpoint identity changed'
    ids = np.load(run / 'test_ids.npy')
    with np.load(run / 'evaluation.npz') as source:
        np.testing.assert_array_equal(ids, source['node_ids'])
        labels = source['y']
    rows = [json.loads(line) for line in (evaluation / 'predictions.jsonl').read_text().splitlines()]
    assert len(rows) == report['nodes'] == len(ids)
    assert len(np.unique(ids)) == len(ids)
    np.testing.assert_array_equal([row['id'] for row in rows], ids)
    decisions = np.asarray([row['sequence_labels'] for row in rows])
    assert decisions.shape == (len(ids), 9)
    assert np.isin(decisions, [0, 1]).all()
    scores = np.asarray([row['score'] for row in rows])
    np.testing.assert_array_equal(scores, decisions.mean(axis=1))
    np.testing.assert_array_equal([row['predicted_label'] for row in rows], scores >= 0.5)
    probabilities = np.asarray([row['llm_probabilities_diagnostic_only'] for row in rows])
    assert probabilities.shape == decisions.shape and np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    non_tie = probabilities != 0.5
    np.testing.assert_array_equal(decisions[non_tie], probabilities[non_tie] > 0.5)
    state = torch.load(evaluation / 'memory.pt', map_location='cpu', weights_only=True)
    assert state['completed_nodes'] == len(ids), 'Predictions extend beyond committed memory'
    measured = measure(labels, scores)
    for key in ['auroc', 'average_precision', 'macro_f1', 'nodes', 'positives', 'predicted_positives']:
        np.testing.assert_allclose(measured[key], report['metrics'][key], rtol=0, atol=1e-12)
    with np.load(run / 'supervision.npz') as source:
        supervised_ids = source['ids']
        supervised_labels = source['y']
        used = np.isin(ids, supervised_ids)
    if held_out:
        assert not used.any(), 'Held-out test labels appeared in the supervision set'
        bank_targets = np.load(run / 'bank/target.npy', mmap_mode='r')
        assert np.isin(bank_targets, supervised_ids).all(), 'Bank contains nontraining centers'
        bank_nodes = np.load(run / 'bank/nodes.npy', mmap_mode='r')
        assert np.isin(bank_nodes[bank_nodes >= 0], supervised_ids).all(), 'Bank contains nontraining nodes'
        with np.load(run / 'evaluation.npz') as source:
            assert not np.isin(source['val_ids'], supervised_ids).any()
        assert data is not None, 'Held-out provenance audit requires the experiment data'
        protocol = json.loads((Path(data) / 'protocol.json').read_text())
        assert protocol.get('test_metrics_used_for_injection_selection') is False
    subsets = {}
    for name, selected in [('test_labels_used_in_training', used), ('test_labels_not_used_in_training', ~used)]:
        if selected.any():
            subsets[name] = measure(labels[selected], scores[selected])
    if data is not None:
        data = Path(data)
        with np.load(data / (prepared['dataset'] + '.npz')) as source:
            np.testing.assert_array_equal(labels, source['y'][ids])
            if held_out:
                np.testing.assert_array_equal(ids, source['test'])
                with np.load(run / 'evaluation.npz') as evaluation_data:
                    np.testing.assert_array_equal(evaluation_data['val_ids'], source['val'])
                np.testing.assert_array_equal(np.sort(supervised_ids), np.sort(source['train']))
                np.testing.assert_array_equal(supervised_labels, source['y'][supervised_ids])
                np.testing.assert_array_equal(np.load(run / 'bank/y.npy'), source['y'][bank_targets])
        with np.load(data / 'injection.npz') as injection:
            for kind in ['structural', 'contextual']:
                selected = (labels == 0) | np.isin(ids, injection[kind + '_nodes'])
                subsets[kind + '_vs_normal'] = measure(labels[selected], scores[selected])
    comparisons = {key: {'paper_target': target, 'measured': measured[key], 'difference_percentage_points': 100 * (measured[key] - target), 'strictly_exceeds': measured[key] > target} for key, target in TARGETS[prepared['dataset']].items()}
    return {'verified_complete': True, 'dataset': prepared['dataset'], 'seed': prepared['seed'], 'run': str(run), 'evaluation': str(evaluation), 'time_unix': time.time(), 'checkpoint_sha256': checkpoint_sha, 'predictions_sha256': digest(evaluation / 'predictions.jsonl'), 'sequence_decisions': int(decisions.size), 'threshold': 0.5, 'decision_mode': config['decision_mode'], 'score_source': 'mean of nine real LLM binary decisions', 'metrics': measured, 'subsets': subsets, 'paper_comparison': comparisons, 'strictly_exceeds_all_three_targets': all((item['strictly_exceeds'] for item in comparisons.values())), 'score_histogram': {str(i): int((decisions.sum(axis=1) == i).sum()) for i in range(10)}, 'label_fraction': prepared['label_fraction'], 'valid_held_out_evaluation': held_out, 'interpretation': 'Training labels and bank centers are confined to the training partition; injection/model selection uses validation results. Synthetic difficulty differs from the unspecified author recipe.' if held_out else 'User-authorized training uses original validation/test labels. Unused-label subsets also do not undo test-informed dataset and model selection.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.evaluation, args.data)
    output = args.output or args.evaluation / 'audit.json'
    temporary = output.with_suffix('.pending.json')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output)
    print(f"Macro-F1: {result['metrics']['macro_f1'] * 100:.4f}%")
if __name__ == '__main__':
    main()
