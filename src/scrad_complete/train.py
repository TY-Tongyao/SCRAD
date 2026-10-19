import argparse
import json
from pathlib import Path
import time
import shutil
import numpy as np
import torch
from scrad.train import seed_all
from .frozen_llm import FrozenBinaryLLM
from .runtime import CompleteRuntime
from .integrity import sha256_stream

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--llm', default='models/DeepSeek-R1-Distill-Qwen-14B-AWQ')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--batch-nodes', type=int, default=2)
    p.add_argument('--micro-batch-nodes', type=int, default=0)
    p.add_argument('--train-views', type=int, default=1, choices=[1, 9])
    p.add_argument('--prompt-format', choices=['verbose', 'shared_paths'], default='shared_paths')
    p.add_argument('--refit-codebook-size', type=int, default=0)
    p.add_argument('--encoder-lr', type=float, default=0.0001)
    p.add_argument('--token-lr', type=float, default=0.001)
    p.add_argument('--positive-fraction', type=float, default=0.5)
    p.add_argument('--refresh-every', type=int, default=100)
    p.add_argument('--selection-window', type=int, default=25)
    p.add_argument('--save-every', type=int, default=25)
    p.add_argument('--vq-weight', type=float, default=0.1)
    p.add_argument('--candidates', type=int, default=0)
    p.add_argument('--max-context', type=int, default=12000)
    p.add_argument('--gpu-memory-fraction', type=float, default=0.25)
    p.add_argument('--keep-activations-on-gpu', action='store_true')
    p.add_argument('--cpu-weight-layers', type=int, default=0)
    p.add_argument('--checkpoint')
    p.add_argument('--resume-from', type=Path)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.checkpoint and args.resume_from:
        raise ValueError('Choose either a weight warm start or an exact training resume')
    if min(args.save_every, args.selection_window, args.refresh_every) < 1:
        raise ValueError('Checkpoint, selection and refresh intervals must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Alignment output is not empty; use a new directory and --checkpoint to continue weights')
    args.output.mkdir(parents=True, exist_ok=True)
    warm_start = None
    if args.checkpoint or args.resume_from:
        warm_start = args.output / 'warm_start.pt'
        if not warm_start.exists():
            shutil.copyfile(args.resume_from or args.checkpoint, warm_start)
    torch.set_num_threads(4)
    seed_all(args.seed)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    cfg = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    prepared = json.loads((args.run / 'prepared.json').read_text())
    cfg.update(llm_weights_frozen=True, loss='binary decoder NLL + straight-through VQ loss', frozen_vocabulary_storage='cpu; exact requested embedding rows transferred to GPU', encoder_score_used_as_prediction=False, valid_held_out_evaluation=prepared.get('valid_held_out_evaluation', False), sequence_labels='inherited from supervised center labels', optimizer_scope='dual edge encoder + VQ codebook + graph-to-LLM embedding map', pmp='frozen supervised warm-start node representations', training_steps_complete=False)
    if warm_start is not None:
        cfg['warm_start_sha256'] = sha256_stream(warm_start)
    (args.output / 'config.json').write_text(json.dumps(cfg, indent=2))
    print('Loading frozen LLM', flush=True)
    llm = FrozenBinaryLLM(args.llm, device='cuda', offload_activations=not args.keep_activations_on_gpu, cpu_weight_layers=args.cpu_weight_layers)
    runtime = CompleteRuntime(args.run, llm, torch.device('cuda'), args.candidates, warm_start, args.prompt_format)
    resumed = None
    if args.resume_from:
        resumed = torch.load(warm_start, map_location='cpu', weights_only=True)
        if resumed.get('format_version', 1) < 2 or not resumed.get('bank_matches_checkpoint_weights'):
            raise ValueError('This checkpoint lacks exact-resume state; use --checkpoint for a weight-only warm start')
        for key in ['batch_nodes', 'train_views', 'positive_fraction', 'encoder_lr', 'token_lr', 'vq_weight', 'prompt_format', 'seed', 'candidates', 'refit_codebook_size', 'refresh_every', 'selection_window', 'save_every']:
            if cfg[key] != resumed['config'][key]:
                raise ValueError(f'Resume changes training semantics: {key}')
    if args.refit_codebook_size and resumed is None:
        print(json.dumps({'stage': 'balanced_codebook_refit', **runtime.refit_balanced_codebook(args.refit_codebook_size, args.seed)}), flush=True)
    with np.load(args.run / 'supervision.npz') as source:
        ids, labels = (source['ids'], source['y'])
    classes = [np.flatnonzero(labels == value) for value in [0, 1]]
    if any((not len(x) for x in classes)):
        raise ValueError('Both supervised classes are required')
    params = list(runtime.dual.parameters()) + list(runtime.tokens.parameters())
    optimizer = torch.optim.Adam([{'params': runtime.dual.parameters(), 'lr': args.encoder_lr}, {'params': runtime.tokens.parameters(), 'lr': args.token_lr}], weight_decay=0.0001)
    rng = np.random.default_rng(args.seed)
    epoch_steps = int(np.ceil(runtime.cfg['corpus_size'] / (args.batch_nodes * args.train_views)))
    total_steps = args.epochs * epoch_steps
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    print(json.dumps({'stage': 'alignment_start', 'steps': total_steps, 'steps_per_epoch': epoch_steps, 'supervised_nodes': len(ids), 'reference_corpus': len(runtime.bank['target']), 'exact_retrieval': args.candidates == 0}), flush=True)
    best = float('inf')
    window = []
    start_step = 0
    if resumed is not None:
        optimizer.load_state_dict(resumed['optimizer'])
        rng.bit_generator.state = resumed['sampler_rng']
        torch.set_rng_state(resumed['torch_rng'])
        torch.cuda.set_rng_state_all(resumed['cuda_rng'])
        best, window = (resumed['selection_best'], resumed['selection_window_values'])
        start_step = resumed['step']
        if total_steps <= start_step:
            raise ValueError('Requested total steps must exceed the resumed completed step')

    def save(step, name):
        saved = {'format_version': 2, 'dual': {k: v.detach().cpu() for k, v in runtime.dual.state_dict().items()}, 'tokens': {k: v.detach().cpu() for k, v in runtime.tokens.state_dict().items()}, 'memory': runtime.retriever.memory.state_dict(), 'config': cfg, 'step': step, 'prepared_config': runtime.cfg, 'optimizer': optimizer.state_dict(), 'sampler_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'selection_best': best, 'selection_window_values': window.copy(), 'bank_matches_checkpoint_weights': True}
        temp = args.output / (name + '.tmp')
        torch.save(saved, temp)
        temp.replace(args.output / name)
    with (args.output / 'history.jsonl').open('w') as history:
        for step in range(start_step + 1, total_steps + 1):
            started = time.monotonic()
            positive_count = int(round(args.batch_nodes * args.positive_fraction))
            if not 0 < positive_count < args.batch_nodes:
                raise ValueError('Batch must contain both supervised classes; increase batch-nodes')
            selected_labels = rng.permutation(np.r_[np.ones(positive_count, np.int64), np.zeros(args.batch_nodes - positive_count, np.int64)])
            chosen = np.array([rng.choice(classes[label]) for label in selected_labels])
            chosen_ids = ids[chosen]
            views = rng.integers(0, 9, size=(len(chosen), 1)) if args.train_views == 1 else None
            optimizer.zero_grad(set_to_none=True)
            micro = args.micro_batch_nodes or args.batch_nodes
            if micro < 1:
                raise ValueError('micro-batch-nodes must be nonnegative')
            value = vq_value = 0.0
            maximum = reachable = 0
            predictions, reference_rows = ([], [])
            for start in range(0, len(chosen), micro):
                stop = min(start + micro, len(chosen))
                prompts, vq_loss, refs, paths = runtime.prompts(chosen_ids[start:stop], training=True, views=views[start:stop] if views is not None else None)
                maximum = max(maximum, max((len(prompt) for prompt in prompts)))
                reachable += paths
                if maximum > args.max_context:
                    raise ValueError(f'Prompt length {maximum} exceeds context limit; no truncation applied')
                (args.output / 'inflight.json').write_text(json.dumps({'step': step, 'micro_start': start, 'node_ids': chosen_ids[start:stop].tolist(), 'prompt_tokens': [len(prompt) for prompt in prompts]}))
                targets = torch.as_tensor(np.repeat(labels[chosen[start:stop], None], args.train_views, axis=1), device='cuda')
                nll, logits = llm.binary_loss(prompts, targets)
                weight = (stop - start) / len(chosen)
                loss = weight * (nll + args.vq_weight * vq_loss)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite frozen-LLM training loss')
                loss.backward()
                predictions.extend(logits[:, 0].detach().argmax(-1).cpu().tolist())
                reference_rows.extend(refs[:, 0])
                value += weight * float(nll.detach())
                vq_value += weight * float(vq_loss.detach())
                del prompts, nll, loss, logits, vq_loss
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            dual_grad = sum((float(p.grad.detach().float().square().sum()) for p in runtime.dual.parameters() if p.grad is not None)) ** 0.5
            token_grad = sum((float(p.grad.detach().float().square().sum()) for p in runtime.tokens.parameters() if p.grad is not None)) ** 0.5
            if any((p.grad is not None for p in llm.parameters())):
                raise RuntimeError('Frozen LLM unexpectedly accumulated parameter gradients')
            optimizer.step()
            first_prediction = np.asarray(predictions)
            for row, correct in enumerate(first_prediction == labels[chosen]):
                runtime.retriever.memory.supervised_feedback(reference_rows[row], bool(correct))
            window.append(value)
            row = {'step': step, 'epoch_fraction': step / epoch_steps, 'nll': value, 'vq_loss': vq_value, 'first_decision_accuracy': float(np.mean(first_prediction == labels[chosen])), 'positive_nodes': int(labels[chosen].sum()), 'max_prompt_tokens': maximum, 'reachable_kg_paths': reachable, 'gradient_norm_before_clip': float(norm), 'dual_gradient_norm': dual_grad, 'token_gradient_norm': token_grad, 'seconds': time.monotonic() - started, 'gpu_peak_gib': torch.cuda.max_memory_allocated() / 2 ** 30}
            history.write(json.dumps(row) + '\n')
            history.flush()
            print(json.dumps(row), flush=True)
            checkpoint_due = step % args.save_every == 0 or step in [1, total_steps]
            selection_due = step % args.selection_window == 0
            if step % args.refresh_every == 0 or checkpoint_due or selection_due:
                runtime.refresh_bank()
            if selection_due:
                average = float(np.mean(window))
                window.clear()
                if average < best:
                    best = average
                    save(step, 'best.pt')
            if checkpoint_due:
                cfg['training_steps_complete'] = step == total_steps
                save(step, 'last.pt')
    (args.output / 'complete.json').write_text(json.dumps({'steps': total_steps, 'epochs_equivalent': total_steps / epoch_steps, 'resumed_from_step': start_step, 'new_steps_this_run': total_steps - start_step, 'full_requested_epochs_completed': total_steps == args.epochs * epoch_steps, 'pilot': bool(args.max_steps), 'llm_weights_frozen': True, 'real_llm_input_gradient': True, 'valid_held_out_evaluation': prepared.get('valid_held_out_evaluation', False)}, indent=2))
if __name__ == '__main__':
    main()
