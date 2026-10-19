import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def main(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['reddit', 'questions'], required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--protocol', choices=['paper', 'compatibility'], default='paper')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--gpu-memory-fraction', type=float)
    parser.add_argument('--limit-nodes', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    devices = args.gpus.split(',')
    if any((not value.isdigit() for value in devices)) or len(set(devices)) != len(devices):
        parser.error('--gpus requires distinct CUDA indices')
    if args.limit_nodes < 0:
        parser.error('--limit-nodes must be nonnegative')
    settings = json.loads((root / 'configs' / (args.dataset + '.json')).read_text())
    config = settings['infer' if args.protocol == 'paper' else 'compatibility_infer']
    args.model = args.model or root / (settings['model'] if args.protocol == 'paper' else config['model'])
    run = (args.run or root / 'artifacts' / args.dataset).resolve()
    candidate = run / 'checkpoint.pt'
    checkpoint = (args.checkpoint or (candidate if candidate.exists() else root / 'weights' / args.dataset / 'checkpoint.pt')).resolve()
    output = (args.output or root / 'outputs' / args.dataset).resolve()
    fraction = args.gpu_memory_fraction if args.gpu_memory_fraction is not None else config['gpu_memory_fraction']
    if not 0 < fraction <= 1:
        parser.error('--gpu-memory-fraction must lie in (0, 1]')
    command = [sys.executable, '-m', 'scrad_complete.infer', '--protocol', args.protocol, '--run', str(run), '--checkpoint', str(checkpoint), '--llm', str(args.model.resolve()), '--output', str(output), '--evaluate', '--decision-mode', 'joint', '--batch-nodes', str(config['batch_nodes']), '--max-batch-tokens', str(config['max_batch_tokens']), '--gpu-memory-fraction', str(fraction), '--limit-nodes', str(args.limit_nodes)]
    if len(devices) > 1:
        command += ['--llm-devices', ','.join((str(i) for i in range(len(devices)))), '--replica-processes']
    if args.dry_run:
        print(json.dumps({'argv': command}))
        return
    for path in [run / 'prepared.json', checkpoint, args.model / 'config.json']:
        if not path.is_file():
            raise FileNotFoundError(path)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = str(root / 'src') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'inference.log').open('a') as log:
        subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads((output / 'report.json').read_text())
    print(f"Macro-F1: {report['metrics']['macro_f1'] * 100:.4f}%")
if __name__ == '__main__':
    main()
