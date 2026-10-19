from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from .data import Partition, require_binary_anomaly_labels
from scrad_complete.integrity import sha256_stream as digest

def load_full_graph(root, dataset):
    root = Path(root)
    with np.load(root / f'{dataset}.npz') as data:
        y = data['y']
        require_binary_anomaly_labels(y, dataset)
        x = data['x']
        splits = {key: data[key] for key in ['train', 'val', 'test']}
    np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(len(y)))
    scaler = StandardScaler().fit(x)
    x = np.clip(scaler.transform(x), -10, 10).astype(np.float32)
    adj = sparse.load_npz(root / f'{dataset}_adj.npz').astype(np.float32).tocsr()
    assert not adj.diagonal().any()
    u, v = sparse.triu(adj, k=1).nonzero()
    edges = np.stack([u, v], axis=1)
    eid = np.arange(len(u)) + 1
    lookup = sparse.csr_matrix((np.r_[eid, eid], (np.r_[u, v], np.r_[v, u])), shape=adj.shape)
    return (Partition(np.arange(len(y)), x, y, adj, edges, lookup), splits, scaler)
