import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['reddit', 'questions'], required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--model', type=Path, default=root / 'models/DeepSeek-R1-Distill-Qwen-14B-AWQ')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--gpu-memory-fraction', type=float)
    parser.add_argument('--stage', choices=['all', 'pretrain', 'prepare', 'single300', 'joint25'], default='all')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.gpu.isdigit():
        parser.error('--gpu requires one CUDA index')
    config = json.loads((root / 'configs' / (args.dataset + '.json')).read_text())
    data = (args.data or root / 'data' / args.dataset).resolve()
    output = (args.output or root / 'runs' / args.dataset).resolve()
    stages = {'pretrain': ('scrad_complete.pretrain', {'dataset': args.dataset, 'data': data, 'output': output / 'pretrain'}), 'prepare': ('scrad_complete.prepare', {'data': data, 'checkpoint': output / 'pretrain/checkpoint.pt', 'output': output / 'prepared'}), 'single300': ('scrad_complete.train', {'run': output / 'prepared', 'llm': args.model.resolve(), 'output': output / 'single300'}), 'joint25': ('scrad_complete.train', {'run': output / 'prepared', 'llm': args.model.resolve(), 'output': output / 'joint25', 'checkpoint': output / 'single300/best.pt'})}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = str(root / 'src') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    for stage, (module, paths) in stages.items():
        if args.stage not in ['all', stage]:
            continue
        values = dict(config[stage], **paths)
        if args.gpu_memory_fraction is not None:
            values['gpu_memory_fraction'] = args.gpu_memory_fraction
        command = [sys.executable, '-m', module]
        for key, value in values.items():
            option = '--' + key.replace('_', '-')
            if value is True:
                command.append(option)
            elif value is not False and value is not None:
                command += [option, str(value)]
        if args.dry_run:
            print(json.dumps({'stage': stage, 'argv': command}))
            continue
        for path in [data / (args.dataset + '.npz'), data / (args.dataset + '_adj.npz')]:
            if not path.is_file():
                raise FileNotFoundError(path)
        output.mkdir(parents=True, exist_ok=True)
        print(stage, flush=True)
        with (output / (stage + '.log')).open('a') as log:
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
if __name__ == '__main__':
    main()
