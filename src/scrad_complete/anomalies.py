import hashlib
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split

def sha256_stream(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def relocate_contextual_nodes(x, source_adj, splits, record, minimum_degree=2, candidates=50, seed=42):
    from .scalable import original_partition_adjacency
    if minimum_degree < 1 or candidates < 1:
        raise ValueError('Positive degree and donor candidate counts are required')
    degree = np.diff(original_partition_adjacency(source_adj, splits).indptr)
    n = len(x)
    membership = np.empty(n, np.int8)
    for group, ids in enumerate(splits.values()):
        membership[ids] = group
    contextual = np.asarray(record['contextual_nodes']).copy()
    original = contextual.copy()
    donors = np.asarray(record['contextual_donors']).copy()
    moved = degree[contextual] < minimum_degree
    occupied = np.zeros(n, bool)
    occupied[record['structural_nodes']] = True
    occupied[contextual[~moved]] = True
    for group, ids in enumerate(splits.values()):
        positions = np.flatnonzero(moved & (membership[original] == group))
        eligible = ids[(degree[ids] >= minimum_degree) & ~occupied[ids]]
        if len(eligible) < len(positions):
            raise ValueError('Not enough eligible contextual nodes in one original partition')
        rng = np.random.default_rng(np.random.SeedSequence([seed, group, 23]))
        selected = rng.choice(eligible, len(positions), replace=False)
        contextual[positions] = selected
        occupied[selected] = True
        for position, target in zip(positions, selected):
            pool = rng.choice(n - 1, min(candidates, n - 1), replace=False)
            pool += pool >= target
            delta = np.asarray(x[pool], np.float64) - np.asarray(x[target], np.float64)
            donors[position] = pool[np.argmax(np.einsum('ij,ij->i', delta, delta))]
    updated = {key: np.array(value, copy=True) for key, value in record.items()}
    updated.update(contextual_nodes=contextual, contextual_donors=donors, original_contextual_nodes=original, relocated_contextual_mask=moved, contextual_source_partition_degree=degree[contextual])
    features = np.array(x, copy=True)
    features[contextual] = x[donors]
    y = np.zeros(n, np.int64)
    y[record['structural_nodes']] = 1
    y[contextual] = 1
    return (features, y, updated)

def inject_anomalies(x, adj, ratio=0.048, structural_fraction=0.5, seed=42, clique_size=20, candidates=50, clique_groups=None):
    if not (0 < ratio < 1 and 0 < structural_fraction < 1):
        raise ValueError('Ratios must be strictly between zero and one')
    if clique_size < 2 or candidates < 1:
        raise ValueError('Clique size >= 2 and candidates >= 1 are required')
    n = len(x)
    if adj.shape != (n, n) or n < 10:
        raise ValueError('Invalid graph dimensions')
    if clique_groups is None:
        count = int(round(n * ratio))
        structural_count = int(round(count * structural_fraction))
    else:
        if clique_groups < 1 or int(clique_groups) != clique_groups:
            raise ValueError('Clique group count must be a positive integer')
        structural_count = clique_size * int(clique_groups)
        count = 2 * structural_count
        if count > n:
            raise ValueError('Not enough nodes for disjoint structural and contextual anomalies')
    if structural_count < 2 or count - structural_count < 1:
        raise ValueError('Insufficient nodes for the requested two anomaly types')
    rng = np.random.default_rng(seed)
    selected = rng.choice(n, count, replace=False)
    structural, contextual = (selected[:structural_count], selected[structural_count:])
    cliques = [structural[i:i + clique_size] for i in range(0, len(structural), clique_size)]
    if len(cliques[-1]) == 1:
        tail = cliques.pop()
        cliques[-1] = np.r_[cliques[-1], tail]
    rows, cols = ([], [])
    for group in cliques:
        a, b = np.triu_indices(len(group), k=1)
        rows.extend([group[a], group[b]])
        cols.extend([group[b], group[a]])
    rows, cols = (np.concatenate(rows), np.concatenate(cols))
    extra = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=adj.shape)
    modified_adj = adj.astype(np.float32).maximum(extra).tocsr()
    modified_adj.sort_indices()
    modified_x = np.array(x, copy=True)
    donors = np.empty(len(contextual), np.int64)
    k = min(candidates, n - 1)
    for i, target in enumerate(contextual):
        pool = rng.choice(n - 1, k, replace=False)
        pool += pool >= target
        delta = np.asarray(x[pool], dtype=np.float64) - np.asarray(x[target], dtype=np.float64)
        donor = pool[np.argmax(np.einsum('ij,ij->i', delta, delta))]
        donors[i] = donor
        modified_x[target] = x[donor]
    y = np.zeros(n, np.int64)
    y[selected] = 1
    train, remaining = train_test_split(np.arange(n), train_size=0.4, random_state=seed, stratify=y)
    val, test = train_test_split(remaining, train_size=1 / 3, random_state=seed, stratify=y[remaining])
    provenance = {'structural_nodes': structural, 'contextual_nodes': contextual, 'contextual_donors': donors, 'clique_offsets': np.r_[0, np.cumsum([len(g) for g in cliques])]}
    return (modified_x, modified_adj, y, {'train': train, 'val': val, 'test': test}, provenance)
