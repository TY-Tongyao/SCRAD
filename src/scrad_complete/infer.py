import argparse
import json
from pathlib import Path
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import numpy as np
import torch
from .metrics import metrics
from .frozen_llm import FrozenBinaryLLM, FrozenBinaryWorker
from .runtime import CompleteRuntime
from .integrity import sha256_stream

def record_inference_config(path, config):
    path = Path(path)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != config:
            resource_keys = {'llm_devices', 'gpu_memory_fraction', 'replica_processes', 'max_batch_tokens', 'group_max_prompts'}
            identity = lambda value: {k: v for k, v in value.items() if k not in resource_keys}
            if identity(previous) != identity(config):
                raise ValueError('Cannot mix different inference configurations in one output directory')
            state_path = path.parent / 'memory.pt'
            completed = torch.load(state_path, map_location='cpu', weights_only=True)['completed_nodes'] if state_path.exists() else 0
            history_path = path.parent / 'resource_changes.json'
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            change = {'time_unix': time.time(), 'completed_nodes': completed, 'before': {k: previous.get(k) for k in sorted(resource_keys)}, 'after': {k: config.get(k) for k in sorted(resource_keys)}, 'checkpoint_sha256': config['checkpoint_sha256'], 'batching_prompts_retrieval_and_decision_rule_unchanged': True}
            history.append(change)
            temporary = history_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(history, indent=2) + '\n')
            temporary.replace(history_path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(config, indent=2))
    temporary.replace(path)

def predict_in_token_budget(llm, prompts, count, max_tokens=0, prefixes=None, replicas=None, group_max_prompts=0):
    if max_tokens <= 0:
        if replicas and len(replicas) > 1:
            raise ValueError('Multiple LLM devices require a positive --max-batch-tokens')
        return llm.predict(prompts, count=count, prefixes=prefixes)
    groups = []
    start = 0
    while start < len(prompts):
        stop, longest = (start, 0)
        while stop < len(prompts):
            if group_max_prompts and stop - start >= group_max_prompts:
                break
            length = max(longest, len(prompts[stop]) + count - 1)
            if length * (stop - start + 1) > max_tokens:
                break
            longest = length
            stop += 1
        if stop == start:
            raise ValueError('One prompt exceeds --max-batch-tokens; increase the budget; no truncation applied')
        groups.append((start, stop))
        start = stop
    replicas = replicas or [llm]
    assignments = [[] for _ in replicas]
    work = [0 for _ in replicas]
    for start, stop in groups:
        worker = min(range(len(replicas)), key=lambda i: work[i])
        assignments[worker].append((start, stop))
        work[worker] += max((len(p) + count - 1 for p in prompts[start:stop])) * (stop - start)

    def run(replica, assigned):
        rows = []
        device = replica.decoder_rows.device
        context = torch.cuda.device(device) if device.type == 'cuda' else nullcontext()
        with context:
            for start, stop in assigned:
                values = prompts[start:stop] if getattr(replica, 'uses_cpu_input', False) else [prompt.to(device) for prompt in prompts[start:stop]]
                d, p = replica.predict(values, count=count, prefixes=prefixes[start:stop] if prefixes is not None else None)
                rows.append((start, d.to(llm.decoder_rows.device), p.to(llm.decoder_rows.device)))
        return rows
    if len(replicas) == 1:
        rows = run(llm, assignments[0])
    else:
        with ThreadPoolExecutor(max_workers=len(replicas)) as pool:
            futures = [pool.submit(run, replica, assigned) for replica, assigned in zip(replicas, assignments)]
            rows = [row for future in futures for row in future.result()]
    rows.sort(key=lambda row: row[0])
    return (torch.cat([row[1] for row in rows]), torch.cat([row[2] for row in rows]))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--llm', default='models/DeepSeek-R1-Distill-Qwen-14B-AWQ')
    p.add_argument('--candidates', type=int, default=0)
    p.add_argument('--batch-nodes', type=int, default=2)
    p.add_argument('--max-batch-tokens', type=int, default=0)
    p.add_argument('--group-max-prompts', type=int, default=0)
    p.add_argument('--limit-nodes', type=int, default=0)
    p.add_argument('--decision-mode', choices=['joint', 'independent'], default='joint')
    p.add_argument('--gpu-memory-fraction', type=float, default=0.25)
    p.add_argument('--cpu-weight-layers', type=int, default=0)
    p.add_argument('--llm-devices')
    p.add_argument('--replica-processes', action='store_true')
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--split', choices=['test', 'val'], default='test')
    p.add_argument('--prefix-cache', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    device_ids = [int(value) for value in args.llm_devices.split(',')] if args.llm_devices else [0]
    if len(set(device_ids)) != len(device_ids):
        raise ValueError('LLM device indices must be distinct')
    if args.replica_processes and len(device_ids) < 2:
        raise ValueError('--replica-processes requires at least two --llm-devices')
    devices = [torch.device(f'cuda:{index}') for index in device_ids]
    for device in devices:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / 'checkpoint_used.pt'
    if not snapshot.exists():
        temporary = snapshot.with_suffix('.pending.pt')
        shutil.copyfile(args.checkpoint, temporary)
        temporary.replace(snapshot)
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.split == 'test':
        cfg.pop('split')
    if args.llm_devices is None:
        cfg.pop('llm_devices')
    if not args.replica_processes:
        cfg.pop('replica_processes')
    cfg['checkpoint_sha256'] = sha256_stream(snapshot)
    cfg['frozen_vocabulary_storage'] = 'cpu; exact requested embedding rows transferred to GPU'
    config_path = args.output / 'config.json'
    record_inference_config(config_path, cfg)
    ids = np.load(args.run / f'{args.split}_ids.npy')
    if args.limit_nodes:
        ids = ids[:args.limit_nodes]
    replicas = [FrozenBinaryLLM(args.llm, device=devices[0], gradient_checkpointing=False, cpu_weight_layers=args.cpu_weight_layers)]
    for device in devices[1:]:
        replica = FrozenBinaryWorker(args.llm, device, args.gpu_memory_fraction, args.cpu_weight_layers) if args.replica_processes else FrozenBinaryLLM(args.llm, device=device, gradient_checkpointing=False, cpu_weight_layers=args.cpu_weight_layers)
        replicas.append(replica)
    llm = replicas[0]
    runtime = CompleteRuntime(args.run, llm, devices[0], args.candidates, snapshot, inference_only=True, evaluation_split=args.split)
    prediction_path = args.output / 'predictions.jsonl'
    complete = []
    if prediction_path.exists():
        lines = prediction_path.read_text().splitlines()
        state_path = args.output / 'memory.pt'
        if state_path.exists():
            state = torch.load(state_path, map_location='cpu', weights_only=True)
            committed = state['completed_nodes']
            if len(lines) < committed:
                raise ValueError('Committed predictions are missing')
            complete = [json.loads(line) for line in lines[:committed]]
            runtime.retriever.memory.load_state_dict(state['memory'])
        trimmed = prediction_path.with_suffix('.resume.tmp')
        trimmed.write_text(''.join((json.dumps(row) + '\n' for row in complete)))
        trimmed.replace(prediction_path)
        np.testing.assert_array_equal([row['id'] for row in complete], ids[:len(complete)])
    with prediction_path.open('a') as output, torch.no_grad():
        for start in range(len(complete), len(ids), args.batch_nodes):
            started = time.monotonic()
            batch = ids[start:start + args.batch_nodes]
            if args.decision_mode == 'joint':
                prompts, _, _, reachable = runtime.prompts(batch)
                prepared_at = time.monotonic()
                decisions, probabilities = predict_in_token_budget(llm, prompts, count=9, max_tokens=args.max_batch_tokens, prefixes=runtime.prompt_prefixes if args.prefix_cache else None, replicas=replicas, group_max_prompts=args.group_max_prompts)
            else:
                repeated = np.repeat(batch, 9)
                views = np.tile(np.arange(9), len(batch))[:, None]
                prompts, _, _, reachable = runtime.prompts(repeated, views=views)
                prepared_at = time.monotonic()
                decisions, probabilities = predict_in_token_budget(llm, prompts, count=1, max_tokens=args.max_batch_tokens, prefixes=runtime.prompt_prefixes if args.prefix_cache else None, replicas=replicas, group_max_prompts=args.group_max_prompts)
                decisions = decisions.reshape(len(batch), 9)
                probabilities = probabilities.reshape(len(batch), 9)
            for node, labels, probs in zip(batch, decisions.cpu().numpy(), probabilities.cpu().numpy()):
                score = float(labels.mean())
                row = {'id': int(node), 'sequence_labels': labels.tolist(), 'score': score, 'predicted_label': int(score >= 0.5), 'llm_probabilities_diagnostic_only': probs.tolist()}
                output.write(json.dumps(row) + '\n')
                complete.append(row)
            output.flush()
            state_path = args.output / 'memory.pt'
            temp = state_path.with_suffix('.tmp')
            torch.save({'completed_nodes': len(complete), 'memory': runtime.retriever.memory.state_dict()}, temp)
            temp.replace(state_path)
            progress = {'completed_nodes': len(complete), 'total_nodes': len(ids), 'seconds': time.monotonic() - started, 'reachable_kg_paths': reachable, 'graph_retrieval_prompt_seconds': prepared_at - started, 'llm_seconds': time.monotonic() - prepared_at, 'prefix_cache_hits': sum((model.prefix_cache_hits for model in replicas)), 'prefix_cache_misses': sum((model.prefix_cache_misses for model in replicas)), 'max_prompt_tokens': max(map(len, prompts))}
            (args.output / 'progress.json').write_text(json.dumps(progress))
    scores = np.array([row['score'] for row in complete])
    report = {'complete': True, 'nodes': len(ids), 'threshold': 0.5, 'score_source': 'mean of nine real LLM binary decisions', 'encoder_score_used_as_prediction': False, 'valid_held_out_evaluation': runtime.cfg.get('valid_held_out_evaluation', False), 'complete_original_test': args.limit_nodes == 0 and args.split == 'test', 'decision_mode': args.decision_mode, 'exact_full_corpus_retrieval': args.candidates == 0}
    if args.split == 'val':
        report.update(evaluation_split='val', complete_evaluation_split=args.limit_nodes == 0)
    if args.evaluate:
        with np.load(args.run / 'evaluation.npz') as data:
            id_key, y_key = ('node_ids', 'y') if args.split == 'test' else ('val_ids', 'val_y')
            np.testing.assert_array_equal(data[id_key][:len(ids)], ids)
            labels = data[y_key][:len(ids)]
        report['metrics'] = metrics(labels, scores)
        report['metrics']['macro_f1'] = report['metrics']['macro_f1_at_0.5']
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    if args.evaluate:
        print(f"Macro-F1: {report['metrics']['macro_f1'] * 100:.4f}%", flush=True)
if __name__ == '__main__':
    main()
