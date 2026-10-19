import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scrad.model import DualEdgeEncoder
from scrad.train import metrics
from .scalable import MappedEdgeScales
from .anomalies import sha256_stream

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--checkpoint', type=Path, help='Diagnose an aligned checkpoint instead of the initial dual encoder')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--gpu-memory-fraction', type=float, default=0.03)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    cfg = json.loads((args.run / 'prepared.json').read_text())
    saved = torch.load(args.run / 'encoder.pt', map_location='cpu', weights_only=True)
    model = DualEdgeEncoder(cfg['hidden'], cfg['global_depth'], cfg['local_hops'], cfg['alpha']).to(device)
    dual_state = saved['dual']
    if args.checkpoint:
        aligned = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if aligned['prepared_config'] != cfg:
            raise ValueError('Alignment checkpoint belongs to a different prepared graph/run')
        dual_state = aligned['dual']
    model.load_state_dict(dual_state)
    model.eval()
    h = torch.as_tensor(np.load(args.run / 'test/node_embeddings.npy'), device=device)
    scales = MappedEdgeScales(np.load(args.run / 'test/global_scales.npy', mmap_mode='r'), device)
    sequences = np.load(args.run / 'test/sequences.npy', mmap_mode='r')
    edges = np.load(args.run / 'test/edge_ids.npy', mmap_mode='r')
    context_ids = np.load(args.run / 'test/node_ids.npy')
    node_ids = np.load(args.run / 'test_ids.npy')
    inverse = np.full(int(context_ids.max()) + 1, -1, np.int64)
    inverse[context_ids] = np.arange(len(context_ids))
    local = inverse[node_ids]
    if (local < 0).any():
        raise ValueError('Test nodes are missing from their graph context')
    rows = (local[:, None] * cfg['sequences_per_node'] + np.arange(cfg['sequences_per_node'])).ravel()
    with torch.no_grad():
        pmp_scores = torch.nn.functional.linear(h[local], saved['pmp']['classifier.weight'].to(device), saved['pmp']['classifier.bias'].to(device)).sigmoid().squeeze(-1).cpu().numpy()
        parts = []
        for start in range(0, len(rows), args.batch_size):
            selected = rows[start:start + args.batch_size]
            seq = torch.tensor(np.asarray(sequences[selected]), device=device, dtype=torch.long)
            eid = torch.tensor(np.asarray(edges[selected]), device=device, dtype=torch.long)
            _, _, logits = model(h, scales, seq, eid)
            parts.append(logits.sigmoid().cpu().numpy())
    scores = np.concatenate(parts).reshape(-1, cfg['sequences_per_node']).mean(1)
    np.savez_compressed(args.output / 'predictions.npz', node_ids=node_ids, pmp_score=pmp_scores, encoder_score=scores)
    with np.load(args.run / 'evaluation.npz') as source:
        np.testing.assert_array_equal(source['node_ids'], node_ids)
        y = source['y']
    report = {'component': 'auxiliary graph heads', 'complete_scrad_llm_result': False, 'dual_checkpoint': str(args.checkpoint or args.run / 'encoder.pt'), 'dual_checkpoint_sha256': sha256_stream(args.checkpoint or args.run / 'encoder.pt'), 'complete_original_test': True, 'threshold': 0.5, 'valid_held_out_evaluation': cfg.get('valid_held_out_evaluation', False), 'pmp': metrics(y, pmp_scores), 'dual_encoder': metrics(y, scores)}
    with np.load(args.run / 'supervision.npz') as source:
        seen = np.isin(node_ids, source['ids'])
    report['test_labels_used_in_training'] = int(seen.sum())
    for name, subset in [('trained_test_nodes', seen), ('untrained_test_nodes', ~seen)]:
        if len(np.unique(y[subset])) == 2:
            report[name] = {'nodes': int(subset.sum()), 'dual_encoder': metrics(y[subset], scores[subset])}
    if args.data:
        with np.load(args.data / 'injection.npz') as injection:
            for kind in ['structural', 'contextual']:
                positive = np.isin(node_ids, injection[kind + '_nodes'])
                subset = (y == 0) | positive
                report[kind + '_vs_normal'] = metrics(y[subset], scores[subset])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
if __name__ == '__main__':
    main()
