import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scrad_complete.protocol import PAPER_MODEL, PAPER_REVISION, MEMORY_RULE, SETTINGS

def load_config(dataset):
    config = json.loads((ROOT / 'configs' / f'{dataset}.json').read_text())
    lr = SETTINGS[dataset]['learning_rate']
    required = {'pretrain': dict(protocol='paper', graph_mode='full', hidden=256, epochs=0, label_fraction=0.0, lr=lr), 'prepare': dict(protocol='paper', graph_mode='full', codebook_size=512, corpus_size=SETTINGS[dataset]['corpus_size'], include_validation=True), 'alignment': dict(protocol='paper', epochs=50, max_steps=0, train_views=9, encoder_lr=lr, token_lr=lr, vq_weight=0.0, refresh_every=1, refit_codebook_size=0, candidates=0), 'infer': dict(protocol='paper', decision_mode='joint', candidates=0)}
    if config.get('protocol') != 'paper' or config.get('model') != PAPER_MODEL:
        raise ValueError('Default training must select the paper protocol and original model')
    for stage, expected in required.items():
        for key, value in expected.items():
            if config.get(stage, {}).get(key) != value:
                raise ValueError(f'Paper configuration requires {stage}.{key}={value!r}')
    if len(config['seeds']) != 5 or len(set(config['seeds'])) != 5:
        raise ValueError('The default configuration requires five distinct seeds')
    return config

def flags(config):
    result = []
    for key, value in config.items():
        if value is True:
            result.append('--' + key.replace('_', '-'))
        elif value is not False and value is not None:
            result.extend(['--' + key.replace('_', '-'), value])
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=sorted(SETTINGS), required=True)
    p.add_argument('--source', '--data', type=Path, default=ROOT / 'data/processed')
    p.add_argument('--output', type=Path)
    p.add_argument('--llm', '--model', type=Path)
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--pmp-epochs', type=int, help='PMP initialization duration is not specified by the paper')
    p.add_argument('--gpu-memory-fraction', type=float, default=0.9)
    p.add_argument('--batch-nodes', type=int, default=2)
    p.add_argument('--gpu', default='0')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    config = load_config(args.dataset)
    args.seeds = args.seeds if args.seeds is not None else config['seeds']
    args.output = args.output or ROOT / 'runs' / args.dataset
    args.llm = args.llm or ROOT / config['model']
    if not args.gpu.isdigit() or not 0 < args.gpu_memory_fraction <= 1 or args.batch_nodes < 1:
        p.error('Require a CUDA index, memory fraction in (0, 1], and positive batch size')
    if args.pmp_epochs is not None:
        if args.pmp_epochs < 1:
            p.error('--pmp-epochs must be positive')
        config['pretrain']['pmp_epochs'] = args.pmp_epochs
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        p.error('Provide distinct training seeds')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Use a fresh output directory to keep old experiments intact')
    if not args.dry_run:
        model_config = args.llm / 'config.json'
        if not model_config.exists():
            p.error('Supply the local official deepseek-ai/DeepSeek-R1-Distill-Qwen-14B checkpoint with --llm')
        if json.loads(model_config.read_text()).get('quantization_config'):
            p.error('The paper workflow requires unquantized weights; the historical AWQ model is a different numerical setting')
    out = args.output.resolve()
    data = out / 'data'
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')

    def run(arguments):
        command = [sys.executable, *map(str, arguments)]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    run(['scripts/generate_anomalies.py', '--dataset', args.dataset, '--protocol', 'paper', '--source', args.source.resolve(), '--output', data])
    gpu = ['--gpu-memory-fraction', args.gpu_memory_fraction]
    reports = []
    for seed in args.seeds:
        folder = out / f'seed_{seed}'
        run(['-m', 'scrad_complete.pretrain', *flags(config['pretrain']), '--dataset', args.dataset, '--data', data, '--output', folder / 'pretrain', '--seed', seed, *gpu])
        run(['-m', 'scrad_complete.prepare', *flags(config['prepare']), '--data', data, '--checkpoint', folder / 'pretrain/checkpoint.pt', '--output', folder / 'prepared', *gpu])
        run(['-m', 'scrad_complete.train', *flags(config['alignment']), '--run', folder / 'prepared', '--output', folder / 'alignment', '--llm', args.llm.resolve(), '--seed', seed, '--batch-nodes', args.batch_nodes, *gpu])
        inference = config['infer'] | dict(batch_nodes=args.batch_nodes, gpu_memory_fraction=args.gpu_memory_fraction)
        run(['-m', 'scrad_complete.infer', *flags(inference), '--run', folder / 'prepared', '--checkpoint', folder / 'alignment/last.pt', '--output', folder / 'test', '--llm', args.llm.resolve(), '--evaluate'])
        if not args.dry_run:
            reports.append(json.loads((folder / 'test/report.json').read_text()))
    if reports:
        import numpy as np
        summary = dict(method_revision=PAPER_REVISION, seeds=args.seeds, runs=len(reports), five_runs_completed=len(reports) == 5, metric_scale='0..1', std_ddof=1, memory_rule=MEMORY_RULE, output_convention='0/1 constrained decoder retained by user instruction', appendix_output_reproduced=False)
        summary['metrics'] = {key: dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)) if len(values) > 1 else None) for key in ('auroc', 'auc_pr', 'macro_f1') for values in [[r['metrics'][key] for r in reports]]}
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
if __name__ == '__main__':
    main()
