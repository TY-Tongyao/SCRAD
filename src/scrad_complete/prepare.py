import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from scrad.data import Partition, sample_sequences, line_graph_scales
from scrad.full_graph import load_full_graph, digest
from scrad.model import PMPEncoder, DualEdgeEncoder, sparse_tensor
from scrad.train import seed_all
from .anomalies import sha256_stream
from .scalable import csr_tensor, pmp_unknown, fast_sample_sequences, selected_line_graph_scales, MappedEdgeScales, original_partition_adjacency
from .protocol import PAPER_REVISION, validate_paper_splits

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol', choices=['paper', 'legacy'], default='paper')
    p.add_argument('--data', default='data/processed')
    p.add_argument('--device', default='cuda')
    p.add_argument('--corpus-size', type=int)
    p.add_argument('--include-validation', action='store_true', help='Prepare a separate validation context for model selection')
    p.add_argument('--codebook-size', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--graph-mode', choices=['paper_induced', 'original_partitions', 'full'])
    p.add_argument('--gpu-memory-fraction', type=float, default=0.15)
    args = p.parse_args()
    out = args.output
    if (out / 'prepared.json').exists():
        raise ValueError('Prepared run exists; choose a new directory')
    out.mkdir(parents=True, exist_ok=True)
    seed_all(42)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = saved['config']
    name = cfg['dataset']
    paper = args.protocol == 'paper'
    if args.graph_mode is None:
        args.graph_mode = cfg.get('graph_context', 'paper_induced')
    if paper and (cfg.get('method_revision') != PAPER_REVISION or args.graph_mode != 'full' or cfg['hidden'] != 256 or (args.codebook_size != 512) or (saved['label_fraction'] != 0)):
        raise ValueError('Paper preparation requires a new paper initialization, full graph, 512 x 256 codebook and train-only supervision')
    with np.load(Path(args.data) / f'{name}.npz') as data:
        if 'y' not in data.files or not set(np.unique(data['y']).tolist()) <= {0, 1}:
            raise ValueError('Binary anomaly labels are required; community labels are not anomaly labels')
        if data['x'].shape[1] != saved['pmp_state_dict']['project.0.weight'].shape[1]:
            raise ValueError('Checkpoint feature dimension belongs to a different dataset version')
    for original, expected in saved['data_sha256'].items():
        path = Path(args.data) / Path(original).name
        if sha256_stream(path) != expected:
            raise ValueError(f'Checkpoint/data identity mismatch: {path}; train a new checkpoint for this graph')
    corpus_size = args.corpus_size or (200000 if name == 'reddit' else 100000)
    supervised = saved['training_node_ids'].numpy()
    if args.graph_mode in ['paper_induced', 'original_partitions'] or paper:
        with np.load(Path(args.data) / f'{name}.npz') as data:
            raw = data['x']
            labels = data['y']
            splits = {key: data[key] for key in ['train', 'val', 'test']}
        np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(len(labels)))
        if paper:
            validate_paper_splits(splits, len(labels), supervised)
        scaler = StandardScaler().fit(raw[supervised])
        features = np.clip(scaler.transform(raw), -10, 10).astype(np.float32)
        adj = sparse.load_npz(Path(args.data) / f'{name}_adj.npz').tocsr()
        if args.graph_mode == 'original_partitions':
            adj = original_partition_adjacency(adj, splits)
        part = Partition(np.arange(len(labels)), features, labels, adj, None, None)
        del raw
    else:
        part, splits, scaler = load_full_graph(args.data, name)

    def induced(ids):
        adj = part.adj[ids][:, ids].tocsr()
        u, v = sparse.triu(adj, k=1).nonzero()
        numbers = np.arange(len(u)) + 1
        lookup = sparse.csr_matrix((np.r_[numbers, numbers], (np.r_[u, v], np.r_[v, u])), shape=adj.shape)
        return Partition(ids, part.x[ids], part.y[ids], adj, np.stack([u, v], 1), lookup)
    if paper:
        part = induced(part.ids)
    if args.graph_mode == 'paper_induced':
        contexts = {'train': induced(supervised), 'test': induced(splits['test'])}
    elif args.graph_mode == 'original_partitions':
        contexts = {'train': induced(part.ids), 'test': induced(splits['test'])}
    else:
        contexts = {'train': part, 'test': part}
    if args.include_validation:
        contexts['val'] = induced(splits['val']) if args.graph_mode != 'full' else part
    print(json.dumps({'stage': 'contexts_ready', 'nodes': {key: len(value.ids) for key, value in contexts.items()}}), flush=True)
    model = PMPEncoder(part.x.shape[1], cfg['hidden']).to(device)
    model.load_state_dict(saved['pmp_state_dict'])
    model.eval()
    for context_name, graph in contexts.items():
        print(json.dumps({'stage': 'context_embeddings', 'context': context_name, 'edges': len(graph.edges)}), flush=True)
        folder = out / context_name
        folder.mkdir(exist_ok=True)
        with torch.no_grad():
            node_h, _ = pmp_unknown(model, torch.as_tensor(graph.x, device=device), csr_tensor(graph.adj, device))
        query_ids = (supervised if context_name == 'train' else splits[context_name]) if paper else graph.ids
        local_queries = np.searchsorted(graph.ids, query_ids) if paper else np.arange(len(graph.ids))
        qseq, qeids = fast_sample_sequences(graph, np.repeat(local_queries, cfg['sequences_per_node']), np.random.default_rng({'train': 43, 'test': 44, 'val': 45}[context_name]), cfg['sequence_length'])
        if context_name == 'train':
            rng = np.random.default_rng(42)
            eligible = np.flatnonzero(np.isin(graph.ids, supervised))
            centers = np.resize(rng.permutation(eligible), corpus_size)
            print(json.dumps({'stage': 'sampling_corpus', 'sequences': corpus_size}), flush=True)
            sequences, edge_ids = fast_sample_sequences(graph, centers, rng, cfg['sequence_length'])
            selected = np.unique(np.r_[qeids.ravel(), edge_ids.ravel()])
        else:
            selected = np.unique(qeids)
        selected = selected[selected >= 0]
        shape = (max(1, len(selected)), cfg['global_depth'], cfg['hidden'])
        global_h = np.lib.format.open_memmap(folder / 'global_scales.npy', mode='w+', dtype=np.float32, shape=shape)
        if len(selected):
            selected_line_graph_scales(graph, node_h, selected, cfg['global_depth'], cfg['restart'], device=device, output=global_h)
        else:
            global_h[:] = 0
        global_h.flush()
        qeids = np.where(qeids >= 0, np.searchsorted(selected, qeids), -1).astype(np.int32)
        if context_name == 'train':
            edge_ids = np.where(edge_ids >= 0, np.searchsorted(selected, edge_ids), -1).astype(np.int32)
        np.save(folder / 'node_embeddings.npy', node_h.cpu().numpy())
        np.save(folder / 'global_edge_ids.npy', selected)
        np.save(folder / 'node_ids.npy', graph.ids)
        if paper:
            np.save(folder / 'query_node_ids.npy', query_ids)
        np.save(folder / 'sequences.npy', qseq)
        np.save(folder / 'edge_ids.npy', qeids)
    train = contexts['train']
    h = torch.as_tensor(np.load(out / 'train/node_embeddings.npy'), device=device)
    scales = MappedEdgeScales(np.load(out / 'train/global_scales.npy', mmap_mode='r'), device)
    dual = DualEdgeEncoder(cfg['hidden'], cfg['global_depth'], cfg['local_hops'], cfg['alpha'], paper=paper).to(device)
    dual.load_state_dict(saved['dual_state_dict'])
    dual.eval()
    bank_dir = out / 'bank'
    bank_dir.mkdir(exist_ok=True)
    zmap = np.lib.format.open_memmap(bank_dir / 'z.npy', mode='w+', dtype=np.float32, shape=(corpus_size, 2 * cfg['sequence_length'] - 1, cfg['hidden']))
    mmap = np.lib.format.open_memmap(bank_dir / 'mask.npy', mode='w+', dtype=bool, shape=(corpus_size, 2 * cfg['sequence_length'] - 1))
    with torch.no_grad():
        for offset in range(0, corpus_size, args.batch_size):
            seq = torch.as_tensor(sequences[offset:offset + args.batch_size], device=device, dtype=torch.long)
            eid = torch.as_tensor(edge_ids[offset:offset + args.batch_size], device=device, dtype=torch.long)
            z, m, _ = dual(h, scales, seq, eid)
            zmap[offset:offset + len(seq)] = z.cpu().numpy()
            mmap[offset:offset + len(seq)] = m.cpu().numpy()
    zmap.flush()
    mmap.flush()
    global_nodes = np.where(sequences >= 0, train.ids[sequences.clip(0)], -1)
    for key, value in {'target': train.ids[centers], 'nodes': global_nodes, 'y': train.y[centers], 'edge_ids': edge_ids}.items():
        np.save(bank_dir / f'{key}.npy', value)
    original_train = np.isin(train.ids[centers], splits['train'])
    examples = [np.flatnonzero(original_train & (train.y[centers] == label))[0] for label in [0, 1]]
    np.save(bank_dir / 'examples.npy', np.asarray(examples))
    flat = zmap.reshape(-1, cfg['hidden'])
    valid = np.flatnonzero(mmap.ravel())
    chosen = np.random.default_rng(42).choice(valid, min(100000, len(valid)), replace=False)
    print(json.dumps({'stage': 'codebook', 'vectors': len(chosen)}), flush=True)
    vq = MiniBatchKMeans(n_clusters=args.codebook_size, random_state=42, batch_size=4096, n_init=3, max_iter=100)
    vq.fit(np.asarray(flat[chosen]))
    np.save(out / 'codebook.npy', vq.cluster_centers_)
    np.savez_compressed(out / 'supervision.npz', ids=supervised, y=part.y[supervised])
    np.savez_compressed(out / 'evaluation.npz', node_ids=splits['test'], y=part.y[splits['test']], val_ids=splits['val'], val_y=part.y[splits['val']])
    np.save(out / 'test_ids.npy', splits['test'])
    if args.include_validation:
        np.save(out / 'val_ids.npy', splits['val'])
    torch.save({'pmp': saved['pmp_state_dict'], 'dual': saved['dual_state_dict'], 'mean': torch.as_tensor(scaler.mean_), 'scale': torch.as_tensor(scaler.scale_), 'config': cfg}, out / 'encoder.pt')
    metadata = {'dataset': name, 'seed': 42, 'label_fraction': saved['label_fraction'], 'supervised_nodes': len(supervised), 'corpus_size': corpus_size, 'sequence_length': cfg['sequence_length'], 'sequences_per_node': cfg['sequences_per_node'], 'hidden': cfg['hidden'], 'global_depth': cfg['global_depth'], 'local_hops': cfg['local_hops'], 'alpha': cfg['alpha'], 'initial_encoder_checkpoint': str(args.checkpoint), 'initial_encoder_sha256': digest(args.checkpoint), 'graph_context': args.graph_mode, 'host_mapped_edge_scales': True, 'selected_edges_exact_propagation': True, 'partition_overlap': 'training nodes and labels only; validation/test partitions excluded from supervision and references' if saved['label_fraction'] == 0 else 'selected validation/test centers provide supervised references; sequences retain original partition boundaries' if args.graph_mode == 'original_partitions' else 'selected validation/test nodes are intentionally included in the training partition', 'initial_encoder_training': 'supervised warm start; subsequent alignment uses real frozen LLM NLL', 'valid_held_out_evaluation': saved.get('valid_held_out_evaluation', saved['label_fraction'] == 0), 'time_unix': time.time()}
    if paper:
        metadata.update(method_revision=PAPER_REVISION, codebook_size=args.codebook_size, codebook_frozen=True, initial_encoder_training='PMP supervision only; dual encoder is initialized', partition_overlap='full transductive topology; only training targets supply labels and references')
    (out / 'prepared.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)
if __name__ == '__main__':
    main()
