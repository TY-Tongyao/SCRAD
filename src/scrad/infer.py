import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import requests
from .retrieval import Retriever, shared_path
INSTRUCTION = 'You assess coherence of graph-derived sequences for node anomaly detection.\nEach sequence alternates node and edge vector-quantization codes. The target node\nis at original position 3 of 7 nodes; missing neighbors are omitted, never invented.\nCodes are learned from training embeddings. A code number is a categorical ID: its\nnumeric size does not imply anomaly, distance, rank, or severity. Repeated codes\nindicate similar embeddings. References are training sequences ranked using\nHungarian-aligned cosine similarity modulated by match entropy and memory utility.\nThe provided reference label is 0=coherent, 1=incoherent. Sequence labels in training\ninherit the center node\'s annotation, so a mixed neighborhood alone is not proof\nthat its center is anomalous. Use the two fixed examples and retrieved references\nto decide each target sequence. Do not mechanically mark every unusual code as 1.\nThe training population positive fraction is provided as a prior, not a forced quota.\nSequences with one node have no edge coherence evidence; use their reference codes.\nTrain and test sequences are sampled in disjoint node partitions. Therefore paths\nto training references can be unreachable; do not invent a shared-node KG path.\nReturn ONLY a JSON object {"predictions":[{"id":integer,"labels":[0 or 1,...]},...]}.\nInclude every requested node exactly once, with exactly one binary label for each\nsequence, in the original sequence order. No explanation, no markdown, no reasoning.\nCompact input schema: each reference is [label, alternating_codes, center_code].\nEach target sequence is [alternating_codes, center_code, reference_indices,\nweighted_similarities]. Reference indices address the references array. The same\nreference may occur more than once when distinct training sequences have identical\ncodes and labels. KG path status is stated once for the whole input.\n'

def coded(codes, nodes):
    return {'tokens': ' '.join((('N' if i % 2 == 0 else 'E') + str(int(c)) for i, c in enumerate(codes) if c >= 0)), 'center_token': 'N' + str(int(codes[6]))}

def parse_predictions(content, node_ids, s):
    content = re.sub('<think>.*?</think>', '', content, flags=re.S).strip()
    content = re.sub('^```(?:json)?\\s*|\\s*```$', '', content)
    obj = json.loads(content)
    rows = obj.get('predictions')
    if not isinstance(rows, list) or len(rows) != len(node_ids):
        raise ValueError('Wrong prediction count')
    result = {}
    for row in rows:
        ident = row.get('id')
        labels = row.get('labels')
        if isinstance(ident, bool) or not isinstance(ident, int) or ident in result:
            raise ValueError('Invalid or duplicate node ID')
        if not isinstance(labels, list) or len(labels) != s or any((type(x) is not int or x not in (0, 1) for x in labels)):
            raise ValueError('Labels must be exactly binary with expected count')
        result[ident] = labels
    if set(result) != set(node_ids):
        raise ValueError('Unexpected node IDs')
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--split', choices=['val', 'test'], default='test')
    ap.add_argument('--base-url', default='http://127.0.0.1:9007/v1')
    ap.add_argument('--model', default='Qwen3.8-27B')
    ap.add_argument('--output-tag', default='qwen38')
    ap.add_argument('--api-config')
    ap.add_argument('--completion-mode', action='store_true')
    ap.add_argument('--tokenizer-path')
    ap.add_argument('--candidates', type=int, default=128)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--concurrency', type=int, default=2)
    ap.add_argument('--nodes-per-request', type=int, default=4)
    ap.add_argument('--limit-nodes', type=int, default=0)
    ap.add_argument('--retrieval-only', action='store_true')
    ap.add_argument('--node-start', type=int, default=0)
    ap.add_argument('--node-end', type=int, default=0, help='Exclusive range end; zero means all nodes')
    args = ap.parse_args()
    run = Path(args.run)
    config = json.loads((run / 'config.json').read_text())
    s = config['sequences_per_node']
    out = run / (args.output_tag + '_' + args.split)
    out.mkdir(exist_ok=True)
    persisted = vars(args).copy()
    persisted.pop('api_config', None)
    config_path = out / 'inference_config.json'
    if config_path.exists():
        old = json.loads(config_path.read_text())
        for k in ['model', 'base_url', 'candidates', 'topk', 'nodes_per_request', 'limit_nodes', 'completion_mode', 'tokenizer_path', 'node_start', 'node_end']:
            default = 0 if k in ('node_start', 'node_end') else False if k == 'completion_mode' else None
            if old.get(k, default) != persisted[k]:
                raise ValueError(f'Refusing to mix incompatible runs: {k}')
    config_path.write_text(json.dumps(persisted, indent=2))
    bank = np.load(run / 'bank.npz')
    with np.load(run / f'{args.split}_queries.npz') as query_file:
        q = {key: query_file[key] for key in query_file.files}
    nodes = q['target'][::s]
    if args.limit_nodes:
        nodes = nodes[:args.limit_nodes]
    lo, hi = (args.node_start, args.node_end or len(nodes))
    if not 0 <= lo < hi <= len(nodes):
        raise ValueError('Invalid node range')
    if lo % args.nodes_per_request:
        raise ValueError('Range start must preserve original request batch boundaries')
    if hi != len(nodes) and hi % args.nodes_per_request:
        raise ValueError('Range end must preserve original request batch boundaries')
    selected_nodes = nodes[lo:hi]
    n = len(nodes) * s
    assert not set(nodes.tolist()) & set(bank['target'].tolist())
    assert not set(q['nodes'][:n][q['nodes'][:n] >= 0].tolist()) & set(bank['nodes'][bank['nodes'] >= 0].tolist())
    retrieval_path = out / 'retrieval.npz'
    if retrieval_path.exists():
        r = np.load(retrieval_path)
        refs, scores = (r['refs'], r['scores'])
        if len(refs) != n:
            raise ValueError('Retrieval cache size mismatch')
    else:
        retriever = Retriever(bank, args.candidates, args.topk)
        ids = []
        sc = []
        start = time.time()
        for offset in range(0, n, 128):
            a, b = retriever.query_batch(q['z'][offset:min(offset + 128, n)], q['mask'][offset:min(offset + 128, n)])
            ids.append(a)
            sc.append(b)
            if offset % 1280 == 0:
                print(json.dumps({'stage': 'retrieval', 'sequences': min(offset + 128, n), 'total': n, 'seconds': time.time() - start}), flush=True)
        refs, scores = (np.concatenate(ids), np.concatenate(sc))
        np.savez_compressed(retrieval_path, refs=refs, scores=scores, frequency=retriever.frequency, usefulness=retriever.utility, last=retriever.last)
    if args.retrieval_only:
        return
    bank_codes = bank['codes']
    bank_nodes = bank['nodes']
    bank_y = bank['y']
    qcodes = q['codes']
    qnodes = q['nodes']
    fixed = [int(np.flatnonzero(bank_y == label)[0]) for label in (0, 1)]
    examples = [dict(coded(bank_codes[i], bank_nodes[i]), label=int(bank_y[i])) for i in fixed]

    def prompt(start, end):
        rid = sorted(set(refs[start * s:end * s].ravel().tolist()))
        refs_list = []
        mapping = {}
        dedup = {}
        for i in rid:
            code = coded(bank_codes[i], bank_nodes[i])
            entry = (int(bank_y[i]), code['tokens'], code['center_token'])
            if entry not in dedup:
                dedup[entry] = len(refs_list)
                refs_list.append(entry)
            mapping[i] = dedup[entry]
        items = []
        for off in range(start, end):
            sequences = []
            for j in range(s):
                i = off * s + j
                code = coded(qcodes[i], qnodes[i])
                seq = [code['tokens'], code['center_token'], [mapping[int(r)] for r in refs[i]], [round(float(x), 3) for x in scores[i]]]
                sequences.append(seq)
            items.append({'id': int(nodes[off]), 'sequences': sequences})
        data = {'training_positive_fraction': float(bank_y.mean()), 'two_shot_examples': examples, 'references': refs_list, 'kg_paths': 'All unreachable: disjoint node partitions', 'targets': items}
        return INSTRUCTION + json.dumps(data, separators=(',', ':'))
    completed = {}
    pred_path = out / 'predictions.jsonl'
    if pred_path.exists():
        for line in pred_path.read_text().splitlines():
            row = json.loads(line)
            completed[row['id']] = row
    if not set(completed).issubset(set(selected_nodes.tolist())):
        raise ValueError('Cached predictions fall outside the requested node range')
    headers = {}
    if args.api_config:
        secret = json.loads(Path(args.api_config).read_text())
        headers['Authorization'] = 'Bearer ' + secret['api_key']
    tokenizer = None
    if args.completion_mode:
        if not args.tokenizer_path:
            raise ValueError('--completion-mode needs a local --tokenizer-path')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)

    def call(start, end):
        p = prompt(start, end)
        wanted = nodes[start:end].tolist()
        payload = {'model': args.model, 'messages': [{'role': 'user', 'content': p}], 'temperature': 0, 'max_tokens': max(512, len(wanted) * s * 6 + 256), 'chat_template_kwargs': {'enable_thinking': False}}
        endpoint = '/chat/completions'
        if tokenizer is not None:
            p = tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
            if not p.endswith('<think>\n'):
                p += '<think>\n'
            p += '\n</think>\n'
            schema = {'type': 'object', 'properties': {'predictions': {'type': 'array', 'minItems': len(wanted), 'maxItems': len(wanted), 'items': {'type': 'object', 'properties': {'id': {'type': 'integer', 'enum': wanted}, 'labels': {'type': 'array', 'minItems': s, 'maxItems': s, 'items': {'type': 'integer', 'enum': [0, 1]}}}, 'required': ['id', 'labels'], 'additionalProperties': False}}}, 'required': ['predictions'], 'additionalProperties': False}
            payload = {'model': args.model, 'prompt': p, 'temperature': 0, 'max_tokens': max(512, len(wanted) * s * 6 + 256), 'guided_json': schema}
            if len(tokenizer.encode(p, add_special_tokens=False)) + payload['max_tokens'] > 16384:
                raise ValueError('Prompt exceeds deployment context; use a smaller nodes-per-request in a new output tag')
            endpoint = '/completions'
        for attempt in range(4):
            begin = time.time()
            try:
                response = requests.post(args.base_url.rstrip('/') + endpoint, json=payload, headers=headers, timeout=180)
                response.raise_for_status()
                raw = response.json()
                content = raw['choices'][0]['text'] if tokenizer is not None else raw['choices'][0]['message']['content']
                parsed = parse_predictions(content, wanted, s)
                return {'start': start, 'end': end, 'predictions': parsed, 'prompt_sha256': hashlib.sha256(p.encode()).hexdigest(), 'prompt': p, 'response': raw, 'seconds': time.time() - begin, 'attempt': attempt + 1}
            except (requests.RequestException, ValueError, KeyError) as error:
                if attempt == 3:
                    raise RuntimeError(f'Failed batch {start}:{end}: {type(error).__name__}') from error
                time.sleep(min(2 ** attempt, 8))
        raise AssertionError('unreachable')
    tasks = [(off, min(off + args.nodes_per_request, hi)) for off in range(lo, hi, args.nodes_per_request) if not all((int(x) in completed for x in nodes[off:off + args.nodes_per_request]))]
    done = len(completed)
    start_time = time.time()
    errors = []
    with pred_path.open('a') as pf, (out / 'requests.jsonl').open('a') as rf, ThreadPoolExecutor(args.concurrency) as pool:
        pending = iter(tasks)
        futures = {}

        def submit_one():
            item = next(pending, None)
            if item is not None:
                futures[pool.submit(call, *item)] = item
        for _ in range(args.concurrency):
            submit_one()
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for f in finished:
                batch = futures.pop(f)
                try:
                    result = f.result()
                    rf.write(json.dumps(result) + '\n')
                    rf.flush()
                    for ident, labels in result['predictions'].items():
                        if ident in completed:
                            continue
                        row = {'id': ident, 'sequence_labels': labels, 'score': float(np.mean(labels)), 'model': args.model, 'prompt_sha256': result['prompt_sha256']}
                        pf.write(json.dumps(row) + '\n')
                        completed[ident] = row
                        done += 1
                    pf.flush()
                    print(json.dumps({'stage': 'llm', 'done': done, 'total': len(selected_nodes), 'seconds': time.time() - start_time}), flush=True)
                except Exception as e:
                    errors.append({'batch': batch, 'error': str(e)})
                    print(json.dumps(errors[-1]), flush=True)
                submit_one()
            (out / 'progress.json').write_text(json.dumps({'done': done, 'total': len(selected_nodes), 'errors': errors}, indent=2))
    if errors:
        raise RuntimeError(f'{len(errors)} failed batches; rerun resumes missing predictions')
    from .train import metrics
    ev = np.load(run / f'{args.split}_evaluation.npz')
    assert np.array_equal(ev['node_ids'][:len(nodes)], nodes)
    pred = np.array([completed[int(i)]['score'] for i in selected_nodes])
    report = metrics(ev['y'][lo:hi], pred)
    report.update({'model': args.model, 'complete_test_partition': args.limit_nodes == 0 and lo == 0 and (hi == len(nodes)), 'seed': config['seed'], 'node_range': [lo, hi], 'sequence_count_per_node': s, 'retrieval_candidate_count': args.candidates})
    (out / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
if __name__ == '__main__':
    main()
