import numpy as np
PAPER_REVISION = 'scrad_pr_20261019_r2'
MEMORY_RULE = 'eq19_full_weight_decay'
PAPER_MODEL = 'models/DeepSeek-R1-Distill-Qwen-14B'
SETTINGS = {'reddit': dict(corpus_size=200000, learning_rate=0.001, clique_groups=373), 'questions': dict(corpus_size=100000, learning_rate=0.01, clique_groups=119)}

def validate_paper_splits(splits, size, supervised=None):
    arrays = [np.asarray(splits[name]) for name in ('train', 'val', 'test')]
    combined = np.concatenate(arrays)
    if not np.issubdtype(combined.dtype, np.integer) or not np.array_equal(np.sort(combined), np.arange(size)):
        raise ValueError('Train/val/test IDs must be disjoint and cover the graph')
    if any((abs(len(ids) - fraction * size) > 1.01 for ids, fraction in zip(arrays, (0.4, 0.2, 0.4)))):
        raise ValueError('The paper requires a 40/20/40 target-node split')
    if supervised is not None and (not np.array_equal(np.sort(supervised), np.sort(arrays[0]))):
        raise ValueError('Paper supervision must contain training targets only; validation/test labels are excluded')

def require_paper_prepared(config):
    required = dict(method_revision=PAPER_REVISION, graph_context='full', hidden=256, codebook_size=512, codebook_frozen=True, label_fraction=0.0, sequence_length=7, sequences_per_node=9, valid_held_out_evaluation=True)
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f'Paper alignment requires {key}={expected!r}; regenerate initialization/prepared inputs')

def corpus_epoch_batches(targets, batch_nodes, max_views, rng):
    if batch_nodes < 1 or max_views < 1:
        raise ValueError('Batch and view counts must be positive')
    targets = np.asarray(targets)
    order = np.argsort(targets, kind='stable')
    boundaries = np.r_[0, np.flatnonzero(np.diff(targets[order])) + 1, len(order)]
    groups = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        rows = rng.permutation(order[lo:hi])
        groups.extend((rows[i:i + max_views] for i in range(0, len(rows), max_views)))
    shuffled = rng.permutation(len(groups))
    return [[groups[i] for i in shuffled[start:start + batch_nodes]] for start in range(0, len(shuffled), batch_nodes)]
