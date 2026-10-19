import argparse
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve, auc
from .data import load_partitions, sample_sequences, line_graph_scales
from .model import PMPEncoder, DualEdgeEncoder, sparse_tensor

def metrics(y, score):
    p, r, _ = precision_recall_curve(y, score)
    return {'auroc': float(roc_auc_score(y, score)), 'average_precision': float(average_precision_score(y, score)), 'auc_pr_trapezoid': float(auc(r, p)), 'macro_f1_at_0.5': float(f1_score(y, score >= 0.5, average='macro')), 'nodes': len(y), 'positives': int(np.sum(y)), 'predicted_positives': int(np.sum(score >= 0.5))}

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pmp_forward(model, part, device, known=None):
    x = torch.as_tensor(part.x, device=device)
    adj = sparse_tensor(part.adj, device)
    if known is None:
        known = torch.full((len(x),), 2, dtype=torch.long, device=device)
    return model(x, adj, known)

def encode(model, h, scales, seq, edges, batch_size, device):
    outputs, masks, logits = ([], [], [])
    model.eval()
    with torch.no_grad():
        for off in range(0, len(seq), batch_size):
            a = torch.as_tensor(seq[off:off + batch_size], device=device, dtype=torch.long)
            b = torch.as_tensor(edges[off:off + batch_size], device=device, dtype=torch.long)
            z, m, l = model(h, scales, a, b)
            outputs.append(z.cpu().numpy())
            masks.append(m.cpu().numpy())
            logits.append(l.sigmoid().cpu().numpy())
    return (np.concatenate(outputs), np.concatenate(masks), np.concatenate(logits))

def fit(args):
    selection = getattr(args, 'checkpoint_selection', 'validation_ap')
    if selection not in {'validation_ap', 'final_epoch'}:
        raise ValueError(selection)
    if args.pmp_epochs < 1 or args.epochs < 1:
        raise ValueError('Training epochs must be positive')
    selected = {'method': selection, 'validation_used_for_selection': selection == 'validation_ap', 'stages': {}}
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    device = torch.device(args.device)
    out = Path(args.output) / args.dataset / f'seed_{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2))
    parts, scaler = load_partitions(args.data, args.dataset)
    train, val = (parts['train'], parts['val'])
    model = PMPEncoder(train.x.shape[1], args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    y = torch.as_tensor(train.y, dtype=torch.float32, device=device)
    positive_weight = (len(y) - y.sum()) / y.sum()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    history = []
    best, best_state = (-1, None)
    tx = torch.as_tensor(train.x, device=device)
    ta = sparse_tensor(train.adj, device)
    for epoch in range(args.pmp_epochs):
        model.train()
        opt.zero_grad()
        hidden = torch.rand(len(y), device=device) < 0.5
        known = torch.where(hidden, 2, y.long())
        h, logit = model(tx, ta, known)
        loss = loss_fn(logit[hidden], y[hidden])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            _, vl = pmp_forward(model, val, device)
            ap = average_precision_score(val.y, vl.sigmoid().cpu().numpy())
        if selection == 'validation_ap' and ap > best or (selection == 'final_epoch' and epoch + 1 == args.pmp_epochs):
            best, best_state = (ap, copy.deepcopy(model.state_dict()))
            selected['stages']['pmp'] = {'epoch': epoch + 1, 'monitored_val_ap': float(ap)}
        row = {'stage': 'pmp', 'epoch': epoch + 1, 'loss': float(loss.detach()), 'val_ap': float(ap)}
        history.append(row)
        if epoch % 5 == 0 or epoch + 1 == args.pmp_epochs:
            print(json.dumps(row), flush=True)
    model.load_state_dict(best_state)
    model.eval()
    embeddings = {}
    pmp_scores = {}
    with torch.no_grad():
        for name, part in parts.items():
            hh, ll = pmp_forward(model, part, device)
            embeddings[name] = hh.cpu().numpy()
            pmp_scores[name] = ll.sigmoid().cpu().numpy()
    torch.save({'state_dict': model.cpu().state_dict(), 'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_}, out / 'pmp.pt')
    del model, tx, ta, opt, h, logit, known
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    seqs = {}
    edge_ids = {}
    centers = {}
    globals_ = {}
    for offset, (name, part) in enumerate(parts.items()):
        rng = np.random.default_rng(42 + offset)
        if name == 'train':
            centers[name] = np.resize(rng.permutation(len(part.ids)), args.train_sequences)
        else:
            centers[name] = np.repeat(np.arange(len(part.ids)), args.sequences_per_node)
        seqs[name], edge_ids[name] = sample_sequences(part, centers[name], rng, args.sequence_length)
        globals_[name] = line_graph_scales(part, embeddings[name], args.global_depth, args.restart)
        if len(globals_[name]) == 0:
            globals_[name] = np.zeros((1, args.global_depth, args.hidden), np.float32)
    dual = DualEdgeEncoder(args.hidden, args.global_depth, args.local_hops, args.alpha).to(device)
    opt = torch.optim.Adam(dual.parameters(), lr=args.lr, weight_decay=0.0001)
    ht = torch.as_tensor(embeddings['train'], device=device)
    gt = torch.as_tensor(globals_['train'], device=device)
    hv = torch.as_tensor(embeddings['val'], device=device)
    gv = torch.as_tensor(globals_['val'], device=device)
    targets = torch.as_tensor(train.y[centers['train']], device=device, dtype=torch.float32)
    rng = np.random.default_rng(args.seed)
    vnode = np.sort(np.random.default_rng(42).choice(len(val.ids), min(1500, len(val.ids)), replace=False))
    vrow = (vnode[:, None] * args.sequences_per_node + np.arange(args.sequences_per_node)).ravel()
    best, best_state = (-1, None)
    for epoch in range(args.epochs):
        dual.train()
        loss_sum = 0.0
        t = time.time()
        order = rng.permutation(len(targets))
        for off in range(0, len(order), args.batch_size):
            idx = order[off:off + args.batch_size]
            sq = torch.as_tensor(seqs['train'][idx], device=device, dtype=torch.long)
            ed = torch.as_tensor(edge_ids['train'][idx], device=device, dtype=torch.long)
            opt.zero_grad()
            _, _, l = dual(ht, gt, sq, ed)
            loss = loss_fn(l, targets[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dual.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach()) * len(idx)
        _, _, vp = encode(dual, hv, gv, seqs['val'][vrow], edge_ids['val'][vrow], args.batch_size, device)
        ap = average_precision_score(val.y[vnode], vp.reshape(-1, args.sequences_per_node).mean(1))
        if selection == 'validation_ap' and ap > best or (selection == 'final_epoch' and epoch + 1 == args.epochs):
            best, best_state = (ap, copy.deepcopy(dual.state_dict()))
            selected['stages']['dual'] = {'epoch': epoch + 1, 'monitored_val_ap': float(ap)}
        row = {'stage': 'dual', 'epoch': epoch + 1, 'loss': loss_sum / len(targets), 'val_ap': float(ap), 'seconds': time.time() - t}
        history.append(row)
        print(json.dumps(row), flush=True)
        (out / 'history.json').write_text(json.dumps(history, indent=2))
    dual.load_state_dict(best_state)
    torch.save(dual.state_dict(), out / 'dual.pt')
    (out / 'checkpoint_selection.json').write_text(json.dumps(selected, indent=2))
    bank_idx = np.sort(np.random.default_rng(42).choice(len(targets), min(args.memory_capacity, len(targets)), replace=False))
    bz, bm, bp = encode(dual, ht, gt, seqs['train'][bank_idx], edge_ids['train'][bank_idx], args.batch_size, device)
    valid = bz[bm]
    vq = MiniBatchKMeans(n_clusters=args.codebook_size, random_state=args.seed, batch_size=4096, n_init=3, max_iter=100)
    fit_idx = np.random.default_rng(42).choice(len(valid), min(100000, len(valid)), replace=False)
    vq.fit(valid[fit_idx])

    def quantize(z, mask):
        codes = np.full(mask.shape, -1, np.int16)
        codes[mask] = vq.predict(z[mask])
        return codes
    nodes = seqs['train'][bank_idx]
    ids = np.where(nodes >= 0, train.ids[nodes.clip(0)], -1)
    bank_y = train.y[centers['train'][bank_idx]]
    np.savez_compressed(out / 'bank.npz', z=bz, mask=bm, codes=quantize(bz, bm), nodes=ids, target=train.ids[centers['train'][bank_idx]], y=bank_y, utility=((bp >= 0.5) == bank_y).astype(np.float32), codebook=vq.cluster_centers_)
    report = {}
    for name in ['val', 'test']:
        part = parts[name]
        h = torch.as_tensor(embeddings[name], device=device)
        g = torch.as_tensor(globals_[name], device=device)
        z, m, p = encode(dual, h, g, seqs[name], edge_ids[name], args.batch_size, device)
        ids = np.where(seqs[name] >= 0, part.ids[seqs[name].clip(0)], -1)
        np.savez_compressed(out / f'{name}_queries.npz', z=z, mask=m, codes=quantize(z, m), nodes=ids, target=part.ids[centers[name]], centers=centers[name])
        np.savez_compressed(out / f'{name}_evaluation.npz', node_ids=part.ids, y=part.y, pmp_score=pmp_scores[name], encoder_score=p.reshape(-1, args.sequences_per_node).mean(1))
        report[name] = {'pmp_control': metrics(part.y, pmp_scores[name]), 'encoder_control': metrics(part.y, p.reshape(-1, args.sequences_per_node).mean(1))}
    (out / 'encoder_metrics.json').write_text(json.dumps(report, indent=2))
    (out / 'training_complete.json').write_text(json.dumps({'time_unix': time.time(), 'cuda_peak_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0}))
    print(json.dumps(report), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['reddit', 'questions'], required=True)
    ap.add_argument('--data', default='data/processed')
    ap.add_argument('--output', default='results/runs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--gpu-memory-fraction', type=float, default=0.04)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--pmp-epochs', type=int, default=50)
    ap.add_argument('--lr', type=float)
    ap.add_argument('--checkpoint-selection', choices=['validation_ap', 'final_epoch'], default='validation_ap')
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--train-sequences', type=int)
    ap.add_argument('--sequence-length', type=int, default=7)
    ap.add_argument('--sequences-per-node', type=int, default=9)
    ap.add_argument('--global-depth', type=int, default=3)
    ap.add_argument('--local-hops', type=int, default=2)
    ap.add_argument('--restart', type=float, default=0.1)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--memory-capacity', type=int, default=10000)
    ap.add_argument('--codebook-size', type=int, default=256)
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 0.001 if args.dataset == 'reddit' else 0.01
    if args.train_sequences is None:
        args.train_sequences = 200000 if args.dataset == 'reddit' else 100000
    fit(args)
if __name__ == '__main__':
    main()
