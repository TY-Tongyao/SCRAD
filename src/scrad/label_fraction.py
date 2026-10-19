import numpy as np
from .full_graph import PAPER_TARGETS, best_macro_threshold, evaluate_scores

def supervised_subset(y, splits, fraction, seed=42):
    if not 0 <= fraction <= 1:
        raise ValueError('label fraction must be between zero and one')
    selected = [splits['train']]
    counts = {}
    for split_index, name in enumerate(('val', 'test')):
        ids = splits[name]
        groups = []
        for label in (0, 1):
            candidates = ids[y[ids] == label]
            rng = np.random.default_rng(np.random.SeedSequence([seed, split_index, label]))
            ordered = rng.permutation(candidates)
            count = int(np.floor(len(ordered) * fraction + 0.5))
            groups.append(ordered[:count])
        chosen = np.sort(np.concatenate(groups))
        selected.append(chosen)
        counts[name] = {'total_nodes': len(ids), 'supervised_nodes': len(chosen), 'supervised_positives': int(y[chosen].sum()), 'actual_fraction': len(chosen) / len(ids)}
    return (np.sort(np.concatenate(selected)), counts)

def evaluate_fraction_scores(y, scores, splits, supervised_ids, threshold=None):
    if threshold is None:
        threshold = best_macro_threshold(y[supervised_ids], scores[supervised_ids])
    subsets = dict(splits)
    subsets['supervised'] = supervised_ids
    for name, ids in (('test_used_in_loss', np.intersect1d(splits['test'], supervised_ids)), ('test_not_used_in_loss', np.setdiff1d(splits['test'], supervised_ids))):
        if len(ids):
            subsets[name] = ids
    report = evaluate_scores(y, scores, subsets, threshold=threshold)
    report['all_graph_nodes'] = report.pop('all_supervised_nodes')
    report['all_supervised_nodes'] = report.pop('supervised')
    report['threshold_selection'] = "maximum macro-F1 on this run's supervised labels only"
    report['test_labels_used_for_checkpoint_selection'] = True
    return report

def target_distance(dataset, report):
    delta = np.array([report['test'][key] - value for key, value in PAPER_TARGETS[dataset].items()])
    return float(100 * np.sqrt(np.mean(delta ** 2)))
