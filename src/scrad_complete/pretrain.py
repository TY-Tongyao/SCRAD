import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler
import torch
from scrad.data import Partition, require_binary_anomaly_labels
from scrad.label_fraction import supervised_subset
from scrad.model import PMPEncoder, DualEdgeEncoder
from scrad.train import seed_all
from .integrity import sha256_stream
from .scalable import csr_tensor, pmp_unknown, fast_sample_sequences, selected_line_graph_scales, MappedEdgeScales, original_partition_adjacency

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['reddit', 'questions'], required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--label-fraction', type=float, default=0.0)
    p.add_argument('--graph-mode', choices=['paper_induced', 'original_partitions', 'full'], default='original_partitions')
    p.add_argument('--pmp-epochs', type=int, default=50)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--pos-weight', type=float, default=6.0)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--corpus-size', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--gpu-memory-fraction', type=float, default=0.2)
    args = p.parse_args()
    if args.pmp_epochs < 1 or args.epochs < 1:
        raise ValueError('Both pretraining stages require at least one epoch')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cfg.update(sequence_length=7, sequences_per_node=9, global_depth=3, restart=0.1, local_hops=2, alpha=0.5, graph_context=args.graph_mode, initialization='random from scratch', labels_in_forward='all unknown', auxiliary_loss='supervised weighted BCE, not the final LLM objective', checkpoint_selection='minimum supervised training loss', valid_held_out_evaluation=args.label_fraction == 0)
    (args.output / 'config.json').write_text(json.dumps(cfg, indent=2))
    paths = [args.data / f'{args.dataset}{suffix}' for suffix in ['.npz', '_adj.npz']]
    identity = {str(path): sha256_stream(path) for path in paths}
    with np.load(paths[0], allow_pickle=False) as source:
        y = source['y']
        require_binary_anomaly_labels(y, args.dataset)
        splits = {key: source[key] for key in ['train', 'val', 'test']}
        raw_x = source['x']
    ids, usage = supervised_subset(y, splits, args.label_fraction, args.seed)
    scaler = StandardScaler().fit(raw_x[ids])
    graph_ids = ids if args.graph_mode == 'paper_induced' else np.arange(len(y))
    x = np.clip(scaler.transform(raw_x[graph_ids]), -10, 10).astype(np.float32)
    del raw_x
    adj = sparse.load_npz(paths[1]).tocsr().astype(np.float32)
    if args.graph_mode == 'paper_induced':
        adj = adj[ids][:, ids].tocsr()
    elif args.graph_mode == 'original_partitions':
        adj = original_partition_adjacency(adj, splits)
    u, v = sparse.triu(adj, k=1).nonzero()
    number = np.arange(len(u), dtype=np.int32) + 1
    lookup = sparse.csr_matrix((np.r_[number, number], (np.r_[u, v], np.r_[v, u])), shape=adj.shape)
    part = Partition(graph_ids, x, y[graph_ids], adj, np.stack([u, v], 1), lookup)
    eligible = np.flatnonzero(np.isin(graph_ids, ids))
    supervised_index = torch.as_tensor(eligible, device=device)
    tx = torch.as_tensor(x, device=device)
    ta = csr_tensor(adj, device)
    labels = torch.as_tensor(part.y, device=device, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    pmp = PMPEncoder(x.shape[1], args.hidden).to(device)
    optimizer = torch.optim.Adam(pmp.parameters(), lr=args.lr, weight_decay=0.0001)
    best = float('inf')
    best_state = None
    history_path = args.output / 'history.jsonl'

    def record(row):
        with history_path.open('a') as log:
            log.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    for epoch in range(1, args.pmp_epochs + 1):
        started = time.monotonic()
        pmp.train()
        optimizer.zero_grad(set_to_none=True)
        _, logits = pmp_unknown(pmp, tx, ta)
        loss = loss_fn(logits[supervised_index], labels[supervised_index])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pmp.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        value = float(loss.detach())
        if value < best:
            best = value
            best_state = copy.deepcopy(pmp.state_dict())
        record({'stage': 'pmp', 'epoch': epoch, 'loss': value, 'seconds': time.monotonic() - started})
    pmp.load_state_dict(best_state)
    pmp.eval()
    with torch.no_grad():
        h, _ = pmp_unknown(pmp, tx, ta)
    pmp_state = {key: value.detach().cpu() for key, value in pmp.state_dict().items()}
    del pmp, optimizer, ta, tx, logits, loss, best_state
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    corpus_size = args.corpus_size or (200000 if args.dataset == 'reddit' else 100000)
    rng = np.random.default_rng(args.seed)
    centers = np.resize(rng.permutation(eligible), corpus_size)
    seq, edge_ids = fast_sample_sequences(part, centers, rng)
    selected = np.unique(edge_ids)
    selected = selected[selected >= 0]
    scale_path = args.output / 'training_edge_scales.npy'
    array = np.lib.format.open_memmap(scale_path, mode='w+', dtype=np.float32, shape=(max(1, len(selected)), 3, args.hidden))
    if len(selected):
        selected_line_graph_scales(part, h, selected, device=device, output=array)
    else:
        array[:] = 0
    array.flush()
    edge_ids = np.where(edge_ids >= 0, np.searchsorted(selected, edge_ids), -1).astype(np.int32)
    scales = MappedEdgeScales(array, device)
    dual = DualEdgeEncoder(args.hidden, 3, 2, 0.5).to(device)
    optimizer = torch.optim.Adam(dual.parameters(), lr=args.lr, weight_decay=0.0001)
    targets = labels[torch.as_tensor(centers, device=device)]
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        dual.train()
        total = 0.0
        order = rng.permutation(len(seq))
        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            _, _, logits = dual(h, scales, torch.as_tensor(seq[rows], device=device, dtype=torch.long), torch.as_tensor(edge_ids[rows], device=device, dtype=torch.long))
            loss = loss_fn(logits, targets[rows])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dual.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        value = total / len(seq)
        if value < best:
            best = value
            checkpoint = {'format_version': 3, 'config': cfg, 'pmp_state_dict': pmp_state, 'dual_state_dict': {key: tensor.detach().cpu() for key, tensor in dual.state_dict().items()}, 'training_node_ids': torch.as_tensor(ids), 'label_fraction': args.label_fraction, 'label_usage': usage, 'data_sha256': identity, 'selected_dual_epoch': epoch, 'scaler_mean': torch.as_tensor(scaler.mean_), 'scaler_scale': torch.as_tensor(scaler.scale_), 'valid_held_out_evaluation': args.label_fraction == 0, 'test_labels_used_for_checkpoint_selection': usage['test']['supervised_nodes'] > 0, 'checkpoint_selection': 'supervised training loss, including the explicitly selected validation/test labels'}
            torch.save(checkpoint, args.output / 'checkpoint.pending.pt')
            (args.output / 'checkpoint.pending.pt').replace(args.output / 'checkpoint.pt')
        record({'stage': 'dual', 'epoch': epoch, 'loss': value, 'seconds': time.monotonic() - started})
    (args.output / 'complete.json').write_text(json.dumps({'pmp_epochs': args.pmp_epochs, 'dual_epochs': args.epochs, 'supervised_nodes': len(ids), 'corpus_size': corpus_size, 'best_training_loss': best, 'final_llm_evaluation': False}, indent=2))
if __name__ == '__main__':
    main()
