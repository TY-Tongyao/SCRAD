import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from scrad.model import DualEdgeEncoder
from .integrity import sha256_stream
from .runtime import encode_reference_bank
from .scalable import MappedEdgeScales

def export(run, checkpoint, output, device='cpu', include_evaluation=False):
    run, checkpoint, output = map(Path, [run, checkpoint, output])
    if output.exists() and any(output.iterdir()):
        raise ValueError('Inference export output must be empty')
    cfg = json.loads((run / 'prepared.json').read_text())
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['prepared_config'] != cfg:
        raise ValueError('Checkpoint belongs to a different prepared graph')
    device = torch.device(device)
    folder = run / 'train'
    context = {'ids': np.load(folder / 'node_ids.npy'), 'h': torch.as_tensor(np.load(folder / 'node_embeddings.npy'), device=device), 'scales': MappedEdgeScales(np.load(folder / 'global_scales.npy', mmap_mode='r'), device)}
    bank = {path.stem: np.load(path, mmap_mode='r') for path in (run / 'bank').glob('*.npy')}
    context['scales'].prefetch(bank['edge_ids'])
    dual = DualEdgeEncoder(cfg['hidden'], cfg['global_depth'], cfg['local_hops'], cfg['alpha']).to(device)
    dual.load_state_dict(saved['dual'])
    vectors = encode_reference_bank(dual, context, bank, device)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'bank').mkdir()
    (output / 'test').mkdir()
    np.save(output / 'bank/z.npy', vectors)
    for name in ['examples', 'mask', 'nodes', 'target', 'y']:
        shutil.copyfile(run / f'bank/{name}.npy', output / f'bank/{name}.npy')
    for name in ['node_ids', 'node_embeddings', 'global_scales', 'sequences', 'edge_ids']:
        shutil.copyfile(run / f'test/{name}.npy', output / f'test/{name}.npy')
    for name in ['test_ids.npy'] + (['evaluation.npz'] if include_evaluation else []):
        shutil.copyfile(run / name, output / name)
    compact_cfg = dict(cfg)
    metadata_changes = []
    if cfg.get('label_fraction') == 0:
        compact_cfg['partition_overlap'] = 'training nodes and labels only; validation/test partitions excluded from supervision and references'
        metadata_changes.append('partition_overlap description')
    if 'initial_encoder_checkpoint' in compact_cfg:
        compact_cfg['initial_encoder_checkpoint'] = 'not bundled; identified by initial_encoder_sha256'
        metadata_changes.append('initial_encoder_checkpoint local path removed')
    (output / 'prepared.json').write_text(json.dumps(compact_cfg, indent=2) + '\n')
    compact = {name: saved[name] for name in ['dual', 'tokens', 'memory', 'prepared_config']}
    compact['prepared_config'] = compact_cfg
    compact['config'] = {'prompt_format': saved['config']['prompt_format']}
    compact['inference_only'] = True
    torch.save(compact, output / 'checkpoint.pt')
    files = {str(path.relative_to(output)): sha256_stream(path) for path in sorted(output.rglob('*')) if path.is_file()}
    manifest = {'format_version': 1, 'inference_only': True, 'checkpoint_sha256': files['checkpoint.pt'], 'files_sha256': files, 'reference_vectors_match_checkpoint': True, 'source_checkpoint_sha256': sha256_stream(checkpoint), 'source_prepared_sha256': sha256_stream(run / 'prepared.json'), 'descriptive_metadata_changes': metadata_changes, 'frozen_llm_directory': saved['config'].get('llm', 'models/DeepSeek-R1-Distill-Qwen-14B-AWQ'), 'score_source': 'mean of nine real LLM binary decisions', 'threshold': 0.5, 'evaluation_labels_included': include_evaluation, 'valid_held_out_evaluation': bool(cfg.get('valid_held_out_evaluation', False)), 'label_fraction': cfg.get('label_fraction'), 'reference_label_scope': 'training partition only' if cfg.get('label_fraction') == 0 else 'see the source experiment label-usage protocol'}
    (output / 'deployment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--gpu-memory-fraction', type=float, default=0.03)
    p.add_argument('--include-evaluation', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    options = vars(args).copy()
    fraction = options.pop('gpu_memory_fraction')
    if torch.device(args.device).type == 'cuda':
        index = torch.device(args.device).index
        torch.cuda.set_per_process_memory_fraction(fraction, torch.cuda.current_device() if index is None else index)
    manifest = export(**options)
    print(json.dumps({'output': str(args.output), 'files': len(manifest['files_sha256']), 'checkpoint_sha256': manifest['checkpoint_sha256']}), flush=True)
if __name__ == '__main__':
    main()
