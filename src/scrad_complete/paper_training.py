import json
from pathlib import Path
import time
import numpy as np
import torch
from scrad.train import seed_all
from .anomalies import sha256_stream
from .frozen_llm import FrozenBinaryLLM
from .protocol import PAPER_REVISION, MEMORY_RULE, SETTINGS, corpus_epoch_batches, require_paper_prepared
from .runtime import CompleteRuntime

def validate_options(args, prepared):
    require_paper_prepared(prepared)
    rate = SETTINGS[prepared['dataset']]['learning_rate']
    for key in ('encoder_lr', 'token_lr'):
        value = getattr(args, key)
        if value is not None and value != rate:
            raise ValueError(f"The paper requires {key}={rate} on {prepared['dataset']}")
        setattr(args, key, rate)
    if args.vq_weight or args.refit_codebook_size or args.candidates or (args.refresh_every != 1):
        raise ValueError('Paper training uses NLL only, a fixed codebook, exact retrieval and current reference embeddings')
    if args.cpu_weight_layers:
        raise ValueError('Quantized weight streaming does not apply to the paper checkpoint')
    if args.positive_fraction != 0.5:
        raise ValueError('Paper training visits every corpus row once; class-balanced resampling is unavailable')
    if args.checkpoint:
        raise ValueError('Use a fresh paper run or --resume-from; historical weight warm starts do not reproduce this protocol')
    if min(args.epochs, args.batch_nodes, args.save_every, args.selection_window) < 1 or args.max_steps < 0 or args.micro_batch_nodes < 0:
        raise ValueError('Invalid training schedule or batch size')

def train_paper(args):
    prepared = json.loads((args.run / 'prepared.json').read_text())
    validate_options(args, prepared)
    model_cfg = json.loads((Path(args.llm) / 'config.json').read_text())
    if model_cfg.get('quantization_config'):
        raise ValueError('Paper training requires the unquantized deepseek-ai/DeepSeek-R1-Distill-Qwen-14B weights')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty alignment output directory')
    torch.set_num_threads(4)
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(method_revision=PAPER_REVISION, memory_rule=MEMORY_RULE, output_convention='0/1 constrained decoder retained by user instruction', appendix_output_reproduced=False, loss='sequence-mean binary decoder NLL (Eq. 12)', optimizer_scope='dual-level edge encoder and linear embedding projection', codebook_frozen=True, llm_weights_frozen=True, pmp_frozen=True, epoch_definition='each training-reference corpus sequence exactly once', class_resampling=False, valid_held_out_evaluation=True, checkpoint_selection='final requested epoch; partial runs explicitly marked as pilots')
    llm = FrozenBinaryLLM(args.llm, device=device, offload_activations=not args.keep_activations_on_gpu)
    runtime = CompleteRuntime(args.run, llm, device, checkpoint=args.resume_from, prompt_format=args.prompt_format)
    if runtime.tokens.codebook.requires_grad:
        raise RuntimeError('The paper codebook must not require gradients')
    groups = [dict(params=[p for p in runtime.dual.parameters() if p.requires_grad], lr=args.encoder_lr), dict(params=list(runtime.tokens.project.parameters()), lr=args.token_lr)]
    optimizer = torch.optim.Adam(groups)
    parameters = [p for group in groups for p in group['params']]
    rng = np.random.default_rng(args.seed)
    initial_codebook = runtime.tokens.codebook.detach().clone()
    step = seen = epoch = cursor = 0
    batches = None
    if args.resume_from:
        saved = torch.load(args.resume_from, map_location='cpu', weights_only=True)
        if saved.get('format_version') != 3 or saved.get('config', {}).get('method_revision') != PAPER_REVISION:
            raise ValueError('Resume requires a paper-training checkpoint with optimizer and corpus position')
        for key in ('batch_nodes', 'train_views', 'encoder_lr', 'token_lr', 'prompt_format', 'seed', 'run', 'llm'):
            if config[key] != saved['config'][key]:
                raise ValueError(f'Resume changes training semantics: {key}')
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['sampler_rng']
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        step, seen, epoch, cursor = (saved['step'], saved['sequences_seen'], saved['epoch'], saved['cursor'])
        batches = [[rows.numpy() for rows in batch] for batch in saved['epoch_batches']] if saved['epoch_batches'] else None
        if epoch >= args.epochs or (args.max_steps and step >= args.max_steps):
            raise ValueError('Requested training total must exceed the completed checkpoint')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

    def save():
        state = dict(format_version=3, dual=runtime.dual.state_dict(), tokens=runtime.tokens.state_dict(), memory=runtime.retriever.memory.state_dict(), config=config, prepared_config=runtime.cfg, optimizer=optimizer.state_dict(), sampler_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [], step=step, sequences_seen=seen, epoch=epoch, cursor=cursor, epoch_batches=[[torch.as_tensor(rows) for rows in batch] for batch in batches] if batches else [], bank_matches_checkpoint_weights=True)
        temporary = args.output / 'last.pending.pt'
        torch.save(state, temporary)
        temporary.replace(args.output / 'last.pt')
    with (args.output / 'history.jsonl').open('w') as history:
        while epoch < args.epochs and (not args.max_steps or step < args.max_steps):
            if batches is None:
                batches = corpus_epoch_batches(runtime.bank['target'], args.batch_nodes, args.train_views, rng)
                cursor = 0
            batch = batches[cursor]
            sequence_count = sum(map(len, batch))
            micro = args.micro_batch_nodes or args.batch_nodes
            optimizer.zero_grad(set_to_none=True)
            value = 0.0
            correct_count = reachable = 0
            maximum = 0
            feedback = []
            started = time.monotonic()
            for start in range(0, len(batch), micro):
                prompts, labels, refs, paths = runtime.corpus_prompts(batch[start:start + micro])
                maximum = max(maximum, max(map(len, prompts)))
                reachable += paths
                if any((len(prompt) + len(label) - 1 > args.max_context for prompt, label in zip(prompts, labels))):
                    raise ValueError('Training prompt exceeds context limit; no truncation applied')
                nll, _ = llm.binary_loss(prompts, labels)
                weight = sum(map(len, labels)) / sequence_count
                loss = weight * nll
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite frozen-LLM NLL')
                loss.backward()
                value += float(loss.detach())
                with torch.no_grad():
                    for prompt, target, references in zip(prompts, labels, refs):
                        decisions, _ = llm.predict([prompt.detach()], count=len(target))
                        correct = (decisions[0] == target).cpu().tolist()
                        correct_count += sum(correct)
                        feedback.extend(zip(references, correct))
                del prompts, nll, loss
            if any((p.grad is not None for p in llm.parameters())) or runtime.tokens.codebook.grad is not None:
                raise RuntimeError('Frozen LLM/codebook unexpectedly accumulated gradients')
            for p in parameters:
                if p.grad is not None and (not torch.isfinite(p.grad).all()):
                    raise FloatingPointError('Non-finite encoder/projection gradient')
            optimizer.step()
            for references, correct in feedback:
                runtime.retriever.memory.supervised_feedback(references, correct)
            runtime.refresh_bank()
            torch.testing.assert_close(runtime.tokens.codebook, initial_codebook, rtol=0, atol=0)
            step += 1
            seen += sequence_count
            cursor += 1
            if cursor == len(batches):
                epoch += 1
                batches = None
                cursor = 0
            row = dict(step=step, completed_epochs=epoch, sequences_seen=seen, nll=value, sequence_accuracy=correct_count / sequence_count, max_prompt_tokens=maximum, reachable_kg_paths=reachable, seconds=time.monotonic() - started)
            history.write(json.dumps(row) + '\n')
            history.flush()
            print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or batches is None:
                save()
    save()
    report = dict(steps=step, completed_epochs=epoch, sequences_seen=seen, epochs_equivalent=seen / len(runtime.bank['target']), full_requested_epochs_completed=epoch == args.epochs, paper_50_epochs_completed=epoch == 50, pilot=epoch != 50 or len(runtime.bank['target']) != SETTINGS[prepared['dataset']]['corpus_size'], paper_corpus_size=len(runtime.bank['target']) == SETTINGS[prepared['dataset']]['corpus_size'], codebook_frozen=True, real_llm_input_gradient=True, valid_held_out_evaluation=True, checkpoint_sha256=sha256_stream(args.output / 'last.pt'))
    (args.output / 'complete.json').write_text(json.dumps(report, indent=2) + '\n')
