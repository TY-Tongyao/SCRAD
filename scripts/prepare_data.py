import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import zipfile
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from download_data import sha256, FILES
SOURCES = {'reddit': 'https://data.dgl.ai/dataset/reddit.zip', 'questions': 'https://github.com/yandex-research/heterophilous-graphs/blob/main/data/questions.npz'}

def save_arrays(path, **arrays):
    temporary = path.with_name(path.stem + '.pending.npz')
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)

def normalize_adjacency(a):
    raw_records = int(a.nnz)
    a = a.tocsr().astype(np.float32)
    a.sum_duplicates()
    unique_directed = int(a.nnz)
    loops = int(np.count_nonzero(a.diagonal()))
    a.data[:] = 1
    a = a.maximum(a.T)
    a.setdiag(0)
    a.eliminate_zeros()
    a.sort_indices()
    return (a, {'raw_edge_records': raw_records, 'raw_unique_directed_edges': unique_directed, 'raw_self_loops': loops, 'undirected_edges_without_loops': int(a.nnz // 2), 'bidirected_edges_without_loops': int(a.nnz)})

def prepare_reddit(root):
    raw = root / 'data/raw/reddit_graphsage'
    archive = raw / 'reddit.zip'
    expected = FILES['reddit'][2]
    archive_hash = sha256(archive)
    if expected is not None and archive_hash != expected:
        raise ValueError('GraphSAGE Reddit archive checksum differs from the pinned source')
    native = ['reddit_data.npz', 'reddit_graph.npz']
    with zipfile.ZipFile(archive) as z:
        for name in native:
            matches = [info for info in z.infolist() if Path(info.filename).name == name]
            if len(matches) != 1:
                raise ValueError(f'Archive must contain exactly one {name}')
            dest = raw / name
            if not dest.exists():
                tmp = dest.with_suffix('.extracting')
                with z.open(matches[0]) as inp, tmp.open('wb') as out:
                    shutil.copyfileobj(inp, out, 8 * 1024 * 1024)
                os.replace(tmp, dest)
    print(json.dumps({'stage': 'load_reddit', 'archive_sha256': archive_hash}), flush=True)
    with np.load(raw / 'reddit_data.npz', allow_pickle=False) as z:
        x = z['feature'].astype(np.float32)
        y = z['label'].astype(np.int64).ravel()
        node_types = z['node_types'].astype(np.int8).ravel()
    if x.shape != (232965, 602) or y.shape != (232965,) or len(np.unique(y)) != 41:
        raise ValueError(f'Unexpected GraphSAGE Reddit dimensions: {x.shape}, {y.shape}')
    if not np.isfinite(x).all() or set(np.unique(node_types)) != {1, 2, 3}:
        raise ValueError('Invalid GraphSAGE Reddit features or native split')
    splits = {name: np.flatnonzero(node_types == value) for name, value in [('train', 1), ('val', 2), ('test', 3)]}
    print(json.dumps({'stage': 'load_reddit_adjacency'}), flush=True)
    original = sparse.load_npz(raw / 'reddit_graph.npz')
    if original.shape != (len(y), len(y)):
        raise ValueError('Reddit adjacency shape differs from node count')
    print(json.dumps({'stage': 'normalize_reddit_adjacency', 'raw_edge_records': int(original.nnz)}), flush=True)
    a, edge_stats = normalize_adjacency(original)
    del original
    gc.collect()
    report = {'dataset_id': 'graphsage_reddit_232965', 'source': SOURCES['reddit'], 'paper': 'Inductive Representation Learning on Large Graphs', 'authors': ['William L. Hamilton', 'Rex Ying', 'Jure Leskovec'], 'conference': 'NIPS 2017', 'project_url': 'https://snap.stanford.edu/graphsage/', 'sha256': archive_hash, 'nodes': len(y), 'features': x.shape[1], **edge_stats, 'task': 'multiclass_node_classification', 'classes': len(np.unique(y)), 'label_note': 'Native subreddit community classes (0..40), not binary anomaly annotations.', 'anomaly_labels_available': False, 'synthetic_anomalies_injected': False, 'requires_anomaly_protocol_for_scrad': True, 'split_protocol': 'Official GraphSAGE temporal train/validation/test split from node_types 1/2/3.', 'split_seed': None, 'splits': {name: {'nodes': len(ids), 'class_counts': np.bincount(y[ids], minlength=41).tolist()} for name, ids in splits.items()}, 'raw_files': {name: {'path': str((raw / name).relative_to(root)), 'sha256': sha256(raw / name)} for name in native}}
    arrays = dict(x=x, y=y, class_labels=y, native_node_types=node_types, split=node_types - 1, dataset_id=np.array(report['dataset_id']), task=np.array(report['task']), **splits)
    return (arrays, a, report)

def prepare_questions(root):
    raw = root / 'data/raw/questions_original.npz'
    if sha256(raw) != FILES['questions'][2]:
        raise ValueError('Questions checksum differs from the recorded source')
    with np.load(raw, allow_pickle=False) as z:
        x, y = (z['node_features'].astype(np.float32), z['node_labels'].astype(np.int64).ravel())
        src, dst = z['edges'].T
    n = len(y)
    original = sparse.coo_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(n, n))
    a, edge_stats = normalize_adjacency(original)
    if not np.isfinite(x).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError('Invalid Questions features or labels')
    train, rest = train_test_split(np.arange(n), train_size=0.4, stratify=y, random_state=42)
    val, test = train_test_split(rest, train_size=n // 5, stratify=y[rest], random_state=42)
    splits = dict(train=train, val=val, test=test)
    split = np.full(n, -1, dtype=np.int8)
    for i, idx in enumerate(splits.values()):
        split[idx] = i
    report = {'source': SOURCES['questions'], 'sha256': sha256(raw), 'nodes': n, 'features': x.shape[1], **edge_stats, 'positive_labels': int(y.sum()), 'positive_ratio': float(y.mean()), 'synthetic_anomalies_injected': False, 'label_note': 'Native Yandex Q activity status, used as an anomaly proxy in GADBench; not independently verified fraud.', 'split_seed': 42, 'splits': {name: {'nodes': len(idx), 'positive_labels': int(y[idx].sum()), 'isolated_in_partition': int((a[idx][:, idx].getnnz(axis=1) == 0).sum())} for name, idx in splits.items()}}
    return (dict(x=x, y=y, split=split, **splits), a, report)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--dataset', choices=['reddit', 'questions', 'all'], default='all')
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    audit = root / 'results/dataset_audit.json'
    audit.parent.mkdir(parents=True, exist_ok=True)
    reports = json.loads(audit.read_text()) if audit.exists() else {}
    for name in SOURCES if args.dataset == 'all' else [args.dataset]:
        arrays, a, report = prepare_reddit(root) if name == 'reddit' else prepare_questions(root)
        print(json.dumps({'stage': 'save_processed', 'dataset': name, 'nodes': len(arrays['y']), 'edges': int(a.nnz)}), flush=True)
        save_arrays(out / f'{name}.npz', **arrays)
        pending = out / f'{name}_adj.pending.npz'
        sparse.save_npz(pending, a)
        os.replace(pending, out / f'{name}_adj.npz')
        reports[name] = report
        temporary = audit.with_suffix('.pending.json')
        temporary.write_text(json.dumps(reports, indent=2) + '\n')
        os.replace(temporary, audit)
        print(json.dumps(report), flush=True)
        del arrays, a
        gc.collect()
if __name__ == '__main__':
    main()
