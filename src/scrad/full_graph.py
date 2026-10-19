import hashlib
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from .data import Partition, sample_sequences, require_binary_anomaly_labels
from .train import metrics
PAPER_TARGETS = {'reddit': {'auroc': 0.9772, 'average_precision': 0.8945, 'macro_f1': 0.9302}, 'questions': {'auroc': 0.9598, 'average_precision': 0.829, 'macro_f1': 0.911}}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

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

def best_macro_threshold(y, score):
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind='stable')
    s = score[order]
    labels = y[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(y) - 1]
    tp = np.r_[0, np.cumsum(labels)[ends]]
    fp = np.r_[0, ends + 1] - tp
    fn = y.sum() - tp
    tn = len(y) - y.sum() - fp
    macro = tp / np.maximum(2 * tp + fp + fn, 1) + tn / np.maximum(2 * tn + fp + fn, 1)
    thresholds = np.r_[np.nextafter(s[0], np.inf), [(s[e] + s[e + 1]) / 2 if e + 1 < len(s) else np.nextafter(s[e], -np.inf) for e in ends]]
    chosen = int(np.argmax(macro))
    threshold = float(thresholds[chosen])
    assert np.isclose(f1_score(y, score >= threshold, average='macro'), macro[chosen], atol=1e-12)
    return threshold

def evaluate_scores(y, score, splits, threshold=None):
    if threshold is None:
        threshold = best_macro_threshold(y, score)
    result = {'threshold': threshold, 'threshold_selection': 'maximum macro-F1 on all supervised nodes', 'valid_held_out_evaluation': False}
    for name, ids in {'all_supervised_nodes': np.arange(len(y)), **splits}.items():
        row = metrics(y[ids], score[ids])
        row['macro_f1'] = float(f1_score(y[ids], score[ids] >= threshold, average='macro'))
        row['confusion_matrix'] = confusion_matrix(y[ids], score[ids] >= threshold, labels=[0, 1]).tolist()
        row['predicted_positives_at_selected_threshold'] = int(np.sum(score[ids] >= threshold))
        result[name] = row
    return result

def target_status(dataset, report, margin=0.0):
    target = PAPER_TARGETS[dataset]
    values = report['test']
    return {key: values[key] >= value + margin for key, value in target.items()}

def attainment(dataset, report):
    return min((report['test'][key] / value for key, value in PAPER_TARGETS[dataset].items()))

def sequence_pool(part, data_root, dataset, directory, seed, count, length=7):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{dataset}_full_seq{count}_length{length}_seed{seed}.npz'
    graph_hash = digest(Path(data_root) / f'{dataset}_adj.npz')
    sampler_hash = digest(Path(__file__).with_name('data.py'))
    if path.exists():
        with np.load(path) as data:
            assert data['graph_sha256'].item() == graph_hash and data['sampler_sha256'].item() == sampler_hash
            seq, edges = (data['sequences'], data['edge_ids'])
    else:
        centers = np.repeat(np.arange(len(part.ids)), count)
        seq, edges = sample_sequences(part, centers, np.random.default_rng(seed), length)
        np.savez_compressed(path, sequences=seq, edge_ids=edges, graph_sha256=graph_hash, sampler_sha256=sampler_hash)
    np.testing.assert_array_equal(seq[:, length // 2], np.repeat(np.arange(len(part.ids)), count))
    return (seq, edges, path)

def score_sequences(model, embeddings, scales, sequences, edges, batch_size, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            seq = torch.as_tensor(sequences[start:start + batch_size], dtype=torch.long, device=device)
            eid = torch.as_tensor(edges[start:start + batch_size], dtype=torch.long, device=device)
            _, _, logit = model(embeddings, scales, seq, eid)
            scores.append(logit.sigmoid().cpu().numpy())
    return np.concatenate(scores)
