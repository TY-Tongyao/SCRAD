import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['reddit', 'questions'], required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--model', type=Path, default=root / 'models/DeepSeek-R1-Distill-Qwen-14B-AWQ')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--gpu-memory-fraction', type=float)
    args = parser.parse_args()
    devices = args.gpus.split(',')
    if any((not value.isdigit() for value in devices)) or len(set(devices)) != len(devices):
        parser.error('--gpus requires distinct CUDA indices')
    config = json.loads((root / 'configs' / (args.dataset + '.json')).read_text())['infer']
    run = (args.run or root / 'artifacts' / args.dataset).resolve()
    checkpoint = (args.checkpoint or root / 'weights' / args.dataset / 'checkpoint.pt').resolve()
    for path in [run / 'prepared.json', checkpoint, args.model / 'config.json']:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = (args.output or root / 'outputs' / args.dataset).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    sys.path.insert(0, str(root / 'src'))
    command = [str(__file__), '--run', str(run), '--checkpoint', str(checkpoint), '--llm', str(args.model.resolve()), '--output', str(output), '--evaluate', '--decision-mode', 'joint', '--batch-nodes', str(config['batch_nodes']), '--max-batch-tokens', str(config['max_batch_tokens']), '--group-max-prompts', str(config['group_max_prompts']), '--gpu-memory-fraction', str(args.gpu_memory_fraction or config['gpu_memory_fraction'])]
    if len(devices) > 1:
        command += ['--llm-devices', ','.join((str(i) for i in range(len(devices)))), '--replica-processes']
    sys.argv = command
    with (output / 'inference.log').open('a') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        from scrad_complete.infer import main as predict
        predict()
    report = json.loads((output / 'report.json').read_text())
    print(f"Macro-F1: {report['metrics']['macro_f1'] * 100:.4f}%")
if __name__ == '__main__':
    main()
