import numpy as np

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
