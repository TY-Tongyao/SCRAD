from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler

@dataclass
class Partition:
    ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    adj: sparse.csr_matrix
    edges: np.ndarray
    edge_lookup: sparse.csr_matrix

def require_binary_anomaly_labels(y, name):
    labels = np.unique(y)
    if not np.array_equal(labels, np.array([0, 1])):
        raise ValueError(f'{name} has {len(labels)} native classes, not binary anomaly labels. GraphSAGE Reddit (232,965 nodes) supplies community classes. Configure and record an anomaly-generation protocol before SCRAD training; see results/reddit_dataset_switch.json.')

def load_partitions(root, name):
    root = Path(root)
    z = np.load(root / f'{name}.npz')
    require_binary_anomaly_labels(z['y'], name)
    a = sparse.load_npz(root / f'{name}_adj.npz')
    scaler = StandardScaler().fit(z['x'][z['train']])
    x = np.clip(scaler.transform(z['x']), -10, 10).astype(np.float32)
    parts = {}
    for s in ['train', 'val', 'test']:
        idx = z[s]
        adj = a[idx][:, idx].tocsr()
        u, v = sparse.triu(adj, k=1).nonzero()
        edges = np.stack([u, v], 1)
        eids = np.arange(len(u)) + 1
        lookup = sparse.csr_matrix((np.r_[eids, eids], (np.r_[u, v], np.r_[v, u])), shape=adj.shape)
        parts[s] = Partition(idx, x[idx], z['y'][idx], adj, edges, lookup)
    return (parts, scaler)

def sample_sequences(part, centers, rng, length=7):
    assert length % 2 == 1
    mid = length // 2
    seq = np.full((len(centers), length), -1, dtype=np.int32)

    def branch(node, visited, count):
        if count == 0:
            return []
        nbrs = part.adj.indices[part.adj.indptr[node]:part.adj.indptr[node + 1]]
        candidates = [int(n) for n in nbrs if int(n) not in visited]
        if not candidates:
            return []
        nxt = candidates[int(rng.integers(len(candidates)))]
        visited.add(nxt)
        return [nxt] + branch(nxt, visited, count - 1)
    for i, center in enumerate(centers):
        visited = {int(center)}
        left = branch(int(center), visited, mid)
        right = branch(int(center), visited, mid)
        seq[i, mid] = center
        if left:
            seq[i, mid - len(left):mid] = left[::-1]
        if right:
            seq[i, mid + 1:mid + 1 + len(right)] = right
    edges = np.full((len(seq), length - 1), -1, dtype=np.int32)
    valid = (seq[:, :-1] >= 0) & (seq[:, 1:] >= 0)
    edges[valid] = np.asarray(part.edge_lookup[seq[:, :-1][valid], seq[:, 1:][valid]]).ravel() - 1
    assert (edges[valid] >= 0).all()
    return (seq, edges)

def line_graph_scales(part, node_h, depth=3, restart=0.1):
    e = part.edges
    initial = 0.5 * (node_h[e[:, 0]] + node_h[e[:, 1]])
    degree = np.asarray(part.adj.sum(1)).ravel()
    edge_degree = np.maximum(degree[e[:, 0]] + degree[e[:, 1]] - 1, 1)
    inv = (1 / np.sqrt(edge_degree)).astype(np.float32)
    rows = np.r_[e[:, 0], e[:, 1]]
    cols = np.tile(np.arange(len(e)), 2)
    incidence = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(node_h), len(e)))
    cur = initial
    scales = []
    for _ in range(depth):
        weighted = cur * inv[:, None]
        cur = (1 - restart) * (incidence.T @ (incidence @ weighted) - weighted) * inv[:, None] + restart * initial
        scales.append(cur)
    return np.stack(scales, 1).astype(np.float32)
