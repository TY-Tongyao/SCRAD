import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from scrad_complete.anomalies import inject_anomalies, relocate_contextual_nodes, sha256_stream
from scrad_complete.protocol import SETTINGS, PAPER_REVISION

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['reddit', 'questions'], default='reddit')
    p.add_argument('--protocol', choices=['paper', 'legacy'], default='paper')
    p.add_argument('--source', type=Path, default=Path('data/processed'))
    p.add_argument('--output', type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ratio', type=float)
    p.add_argument('--structural-fraction', type=float, default=0.5)
    p.add_argument('--clique-size', type=int, default=15)
    p.add_argument('--clique-groups', type=int)
    p.add_argument('--clique-size-mode', choices=['fixed', 'degree_scaled'], default='fixed')
    p.add_argument('--candidates', type=int, default=50)
    p.add_argument('--contextual-scale', type=float, default=1, help='Multiply the original-to-donor feature difference; 1 is feature replacement')
    p.add_argument('--minimum-contextual-partition-degree', type=int, default=0, help='Optional original within-partition source degree for contextual anomaly placement; 0 disables relocation')
    args = p.parse_args()
    paper = args.protocol == 'paper'
    if paper:
        groups = SETTINGS[args.dataset]['clique_groups']
        if args.seed != 42 or args.clique_size != 15 or args.clique_size_mode != 'fixed' or (args.structural_fraction != 0.5) or (args.contextual_scale != 1) or (args.minimum_contextual_partition_degree != 0) or (args.clique_groups not in (None, groups)) or (args.ratio is not None):
            p.error('Paper injection uses seed 42, 15-node cliques, Table 1 group counts, equal contextual nodes and unscaled replacement; use --protocol legacy for other experiments')
        args.clique_groups = groups
    if not np.isfinite(args.contextual_scale) or args.contextual_scale <= 0:
        p.error('--contextual-scale must be finite and positive')
    if args.minimum_contextual_partition_degree < 0:
        p.error('--minimum-contextual-partition-degree must be nonnegative')
    name = args.dataset
    if args.ratio is None:
        args.ratio = 0.048 if name == 'reddit' else 0.073
    if args.output is None:
        stem = 'graphsage_reddit' if name == 'reddit' else 'questions'
        args.output = Path(f'data/anomaly/{stem}_seed{args.seed}' + ('_paper' if paper else ''))
    source = args.source.resolve()
    output = args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError('Use a separate anomaly directory outside the canonical source directory')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output is not empty; refusing to overwrite a previous experiment')
    files = [source / f'{name}.npz', source / f'{name}_adj.npz']
    hashes = {p.name: sha256_stream(p) for p in files}
    with np.load(files[0], allow_pickle=False) as data:
        x = data['x']
        source_labels = data['y']
        source_dataset_id = str(data['dataset_id']) if 'dataset_id' in data else 'graphsage_reddit_232965' if name == 'reddit' else 'questions_yandex_48921'
        if name == 'reddit':
            if x.shape != (232965, 602) or len(np.unique(source_labels)) != 41:
                raise ValueError('Expected canonical 232,965-node, 602-feature, 41-class GraphSAGE Reddit')
            native = {f'official_{key}': data[key] for key in ['train', 'val', 'test']}
            native['community_labels'] = source_labels
        else:
            if x.shape != (48921, 301) or set(np.unique(source_labels)) != {0, 1}:
                raise ValueError('Expected canonical 48,921-node, 301-feature Questions data')
            native = {f'source_{key}': data[key] for key in ['train', 'val', 'test']}
            native['native_labels'] = source_labels
    adj = sparse.load_npz(files[1]).tocsr()
    if adj.diagonal().any() or (adj != adj.T).nnz:
        raise ValueError('Expected a symmetric canonical graph without self-loops')
    source_degree = np.diff(adj.indptr)
    degree_median = float(np.median(source_degree))
    if args.clique_size_mode == 'degree_scaled':
        args.clique_size = max(args.clique_size, int(2 ** np.ceil(np.log2(max(2, 2 * degree_median)))))
    result = inject_anomalies(x, adj, args.ratio, args.structural_fraction, args.seed, args.clique_size, args.candidates, clique_groups=args.clique_groups)
    modified_x, modified_adj, y, splits, provenance = result
    if args.minimum_contextual_partition_degree:
        previous_counts = {key: int(y[ids].sum()) for key, ids in splits.items()}
        modified_x, y, provenance = relocate_contextual_nodes(x, adj, splits, provenance, args.minimum_contextual_partition_degree, args.candidates, args.seed)
        for key, ids in splits.items():
            if int(y[ids].sum()) != previous_counts[key]:
                raise AssertionError('Contextual relocation changed a partition anomaly count')
    if args.contextual_scale != 1:
        centers = provenance['contextual_nodes']
        donors = provenance['contextual_donors']
        modified_x[centers] = x[centers] + args.contextual_scale * (x[donors] - x[centers])
        if not np.isfinite(modified_x).all():
            raise ValueError('Feature perturbation produced non-finite values')
    if hashes != {p.name: sha256_stream(p) for p in files}:
        raise RuntimeError('Source changed during generation; no experiment was saved')
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f'{name}.npz', x=modified_x, y=y, **splits, **native)
    sparse.save_npz(output / f'{name}_adj.npz', modified_adj)
    np.savez_compressed(output / 'injection.npz', **provenance)
    report = {'dataset_id': source_dataset_id + '_synthetic_anomalies', 'source_directory': str(source), 'source_sha256': hashes, 'generator_sha256': sha256_stream(Path(__file__).resolve().parents[1] / 'src/scrad_complete/anomalies.py'), 'generator_cli_sha256': sha256_stream(__file__), 'parameters': {key: value for key, value in vars(args).items() if key not in ['source', 'output']}, 'nodes': len(y), 'features': x.shape[1], 'anomalies': int(y.sum()), 'actual_anomaly_ratio': float(y.mean()), 'structural_nodes': len(provenance['structural_nodes']), 'contextual_nodes': len(provenance['contextual_nodes']), 'original_directed_edges': adj.nnz, 'injected_directed_edges': modified_adj.nnz - adj.nnz, 'source_degree_median': degree_median, 'source_degree_mean': float(source_degree.mean()), 'structural_scale_selection': 'power of two at least twice the source median degree; minimum requested clique size' if args.clique_size_mode == 'degree_scaled' else 'fixed requested clique size', 'test_metrics_used_for_injection_selection': False, 'split_sizes': {key: len(value) for key, value in splits.items()}, 'split_anomalies': {key: int(y[value].sum()) for key, value in splits.items()}, 'native_labels_and_splits_preserved': True, 'binary_label_source': 'new synthetic injections, independent of the native source labels', 'protocol_origin': 'PyGOD clique and farthest-of-k mechanisms; ratio/allocation/hyperparameters chosen for this experiment', 'paper_protocol_exactly_known': False, 'output_sha256': {p.name: sha256_stream(p) for p in output.glob('*.npz')}, 'created_unix': time.time()}
    if paper:
        report.update(method_revision=PAPER_REVISION, protocol_origin='Manuscript Section 4.1 / Table 1: fixed 15-node cliques, equal disjoint contextual nodes, seed 42', paper_protocol_exactly_known=False, unspecified_choice='Contextual replacement selects the farthest of 50 candidate donors by default; donor count/metric and split stratification are not specified by the paper')
    if args.minimum_contextual_partition_degree:
        report['parameters']['minimum_contextual_partition_degree'] = args.minimum_contextual_partition_degree
        report.update(contextual_nodes_relocated=int(provenance['relocated_contextual_mask'].sum()), candidate_selection_uses_unlabelled_source_degree_only=True, contextual_relocation_preserves_partition_anomaly_counts=True, protocol_origin='Recorded clique/farthest-of-k injections with original within-partition source-degree eligibility; not the recovered author recipe')
    (output / 'protocol.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
if __name__ == '__main__':
    main()
