import json
from pathlib import Path
import numpy as np
import torch
from scrad.model import DualEdgeEncoder
from .kg import SequenceKG
from .memory import EntropyRetriever
from .tokenization import GraphTokenizer, SoftPromptBuilder
from .scalable import MappedEdgeScales
from .anomalies import sha256_stream
from .protocol import PAPER_REVISION, require_paper_prepared

@torch.no_grad()
def encode_reference_bank(dual, context, bank, device, batch_size=1024):
    inverse = np.full(int(context['ids'].max()) + 1, -1, np.int64)
    inverse[context['ids']] = np.arange(len(context['ids']))
    sequences = np.where(bank['nodes'] >= 0, inverse[bank['nodes'].clip(0)], -1)
    previous = bank['z']
    reusable = isinstance(previous, np.ndarray) and (not isinstance(previous, np.memmap)) and (previous.dtype == np.float32) and previous.flags.writeable
    updated = previous if reusable else np.empty(previous.shape, np.float32)
    dual.eval()
    for start in range(0, len(sequences), batch_size):
        seq = torch.as_tensor(sequences[start:start + batch_size], device=device)
        edges = torch.tensor(np.asarray(bank['edge_ids'][start:start + batch_size]), device=device, dtype=torch.long)
        z, _, _ = dual(context['h'], context['scales'], seq, edges)
        updated[start:start + len(seq)] = z.cpu().numpy()
    return updated

class CompleteRuntime:

    def __init__(self, run, llm, device, candidates=0, checkpoint=None, prompt_format=None, inference_only=False, evaluation_split='test'):
        self.run, self.llm, self.device = (Path(run), llm, device)
        if evaluation_split not in ['test', 'val']:
            raise ValueError('Choose test or val as the evaluation split')
        self.evaluation_split = evaluation_split
        self.cfg = json.loads((self.run / 'prepared.json').read_text())
        c = self.cfg
        self.paper = c.get('method_revision') == PAPER_REVISION
        if self.paper:
            require_paper_prepared(c)
            if getattr(llm, 'quantized', False):
                raise ValueError('Paper inference requires the unquantized DeepSeek checkpoint')
            if candidates:
                raise ValueError('The paper requires exact retrieval over the full training-reference corpus')
        deployment_path = self.run / 'deployment.json'
        deployment = json.loads(deployment_path.read_text()) if deployment_path.exists() else None
        if deployment:
            if evaluation_split != 'test':
                raise ValueError('Compact deployment contains the fixed test context only')
            if not inference_only or checkpoint is None:
                raise ValueError('Compact deployment supports inference with its exported checkpoint only')
            if sha256_stream(checkpoint) != deployment['checkpoint_sha256']:
                raise ValueError('Compact reference bank belongs to a different checkpoint')
            for name, expected in deployment['files_sha256'].items():
                path = self.run / name
                if not path.resolve().is_relative_to(self.run.resolve()) or sha256_stream(path) != expected:
                    raise ValueError(f'Compact deployment file mismatch: {name}')
        self.contexts = {}
        for name in ['test'] if deployment else list(dict.fromkeys(['train', evaluation_split])):
            folder = self.run / name
            ids = np.load(folder / 'node_ids.npy')
            query_path = folder / 'query_node_ids.npy'
            query_ids = np.load(query_path) if query_path.exists() else ids
            self.contexts[name] = {'h': torch.as_tensor(np.load(folder / 'node_embeddings.npy'), device=device), 'scales': MappedEdgeScales(np.load(folder / 'global_scales.npy', mmap_mode='r'), device), 'sequences': np.load(folder / 'sequences.npy', mmap_mode='r'), 'edges': np.load(folder / 'edge_ids.npy', mmap_mode='r'), 'ids': ids, 'lookup': {int(node): i for i, node in enumerate(query_ids)}}
        self.bank = {p.stem: np.load(p, mmap_mode='r') for p in (self.run / 'bank').glob('*.npy')}
        if self.paper and (not deployment):
            with np.load(self.run / 'supervision.npz') as supervision:
                supervised = supervision['ids']
                supervised_labels = supervision['y']
            if not np.isin(self.bank['target'], supervised).all():
                raise ValueError('Reference corpus contains a non-training target')
            label_lookup = np.full(int(supervised.max()) + 1, -1, np.int64)
            label_lookup[supervised] = supervised_labels
            if not np.array_equal(self.bank['y'], label_lookup[self.bank['target']]):
                raise ValueError('Reference labels must be inherited from their training targets')
            if not np.array_equal(self.bank['nodes'][:, c['sequence_length'] // 2], self.bank['target']):
                raise ValueError('Reference sequence centers do not match their training targets')
            for split in ('test', 'val'):
                path = self.run / f'{split}_ids.npy'
                if path.exists() and np.intersect1d(supervised, np.load(path)).size:
                    raise ValueError('Evaluation target labels overlap paper supervision')
        if not deployment:
            cached_bytes = self.contexts['train']['scales'].prefetch(self.bank['edge_ids'])
        self.dual = DualEdgeEncoder(c['hidden'], c['global_depth'], c['local_hops'], c['alpha'], paper=self.paper).to(device)
        if checkpoint:
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if saved['prepared_config'] != self.cfg:
                raise ValueError('Alignment checkpoint belongs to a different prepared graph/run')
            codebook = saved['tokens']['codebook'].numpy()
            self.tokens = GraphTokenizer(codebook, llm.hidden_size, legacy=not self.paper).to(device)
            self.dual.load_state_dict(saved['dual'])
            self.tokens.load_state_dict(saved['tokens'])
        else:
            self.tokens = GraphTokenizer(np.load(self.run / 'codebook.npy'), llm.hidden_size, legacy=not self.paper).to(device)
            saved = torch.load(self.run / 'encoder.pt', map_location='cpu', weights_only=True)
            self.dual.load_state_dict(saved['dual'])
        if self.paper and tuple(self.tokens.codebook.shape) != (512, 256):
            raise ValueError('The paper requires an actual 512 x 256 frozen codebook')
        self.kg = SequenceKG(self.bank['nodes'], self.bank['target'])
        self.retriever = EntropyRetriever(self.bank, candidates=candidates, topk=5, capacity=10000, device=str(device))
        if checkpoint and 'memory' in saved:
            self.retriever.memory.load_state_dict(saved['memory'])
        if prompt_format is None:
            prompt_format = saved.get('config', {}).get('prompt_format', 'verbose')
        self.builder = SoftPromptBuilder(llm, self.tokens, self.bank, self.kg, prompt_format)
        if checkpoint and (not deployment):
            self.refresh_bank()

    def refit_balanced_codebook(self, size, seed=42):
        if self.paper:
            raise ValueError('The paper codebook is initialized once by K-means and remains fixed')
        from sklearn.cluster import MiniBatchKMeans
        rng = np.random.default_rng(seed)
        rows = []
        for label in [0, 1]:
            eligible = np.flatnonzero(self.bank['y'] == label)
            selected = rng.choice(eligible, min(5000, len(eligible)), replace=False)
            values = self.bank['z'][selected][self.bank['mask'][selected]]
            chosen = rng.choice(len(values), min(50000, len(values)), replace=False)
            rows.append(values[chosen])
        count = min(map(len, rows))
        vectors = np.concatenate([row[:count] for row in rows])
        model = MiniBatchKMeans(n_clusters=size, random_state=seed, n_init=1, batch_size=4096, max_iter=100)
        model.fit(vectors)
        self.tokens.codebook = torch.nn.Parameter(torch.as_tensor(model.cluster_centers_, device=self.device))
        return {'codebook_size': size, 'fit_vectors_per_supervised_class': count}

    @torch.no_grad()
    def refresh_bank(self, batch_size=1024):
        context = self.contexts['train']
        self.bank['z'] = encode_reference_bank(self.dual, context, self.bank, self.device, batch_size)
        state = self.retriever.memory.state_dict()
        self.retriever.update_vectors(self.bank['z'])
        self.retriever.memory.load_state_dict(state)
        self.builder.bank = self.bank

    def prompts(self, ids, training=False, views=None):
        self.dual.train(training)
        self.tokens.train(training)
        context = self.contexts['train' if training else self.evaluation_split]
        count = self.cfg['sequences_per_node']
        local = np.asarray([context['lookup'][int(i)] for i in ids])
        if views is None:
            views = np.broadcast_to(np.arange(count), (len(ids), count))
        else:
            views = np.asarray(views).reshape(len(ids), -1)
        per_node = views.shape[1]
        rows = (local[:, None] * count + views).ravel()
        seq = np.asarray(context['sequences'][rows])
        edge = np.asarray(context['edges'][rows])
        global_nodes = np.where(seq >= 0, context['ids'][seq.clip(0)], -1)
        z, mask, _ = self.dual(context['h'], context['scales'], torch.as_tensor(seq, device=self.device, dtype=torch.long), torch.as_tensor(edge, device=self.device, dtype=torch.long))
        refs, scores = self.retriever.query_batch(z.detach().cpu().numpy(), mask.cpu().numpy(), excluded_centers=np.repeat(ids, per_node))
        prompts, losses = ([], [])
        self.prompt_prefixes = []
        reachable = 0
        for i, center in enumerate(ids):
            start = i * per_node
            stop = start + per_node
            prompt, vq, paths = self.builder.build(z[start:stop], mask[start:stop], global_nodes[start:stop], int(center), refs[start:stop], scores[start:stop])
            prompts.append(prompt)
            losses.append(vq)
            reachable += paths
            self.prompt_prefixes.append(self.builder.last_prefix)
        return (prompts, torch.stack(losses).mean(), refs.reshape(len(ids), per_node, -1), reachable)

    def corpus_prompts(self, groups):
        context = self.contexts['train']
        self.dual.train()
        self.tokens.train()
        if 'inverse' not in context:
            inverse = np.full(int(context['ids'].max()) + 1, -1, np.int64)
            inverse[context['ids']] = np.arange(len(context['ids']))
            context['inverse'] = inverse
        inverse = context['inverse']
        prompts, references, labels = ([], [], [])
        reachable = 0
        for rows in groups:
            centers = np.asarray(self.bank['target'][rows])
            if not len(centers) or not np.all(centers == centers[0]):
                raise ValueError('Each training prompt must contain views of one training target')
            nodes = np.asarray(self.bank['nodes'][rows])
            seq = np.where(nodes >= 0, inverse[nodes.clip(0)], -1)
            edges = np.asarray(self.bank['edge_ids'][rows])
            z, mask, _ = self.dual(context['h'], context['scales'], torch.as_tensor(seq, device=self.device), torch.as_tensor(edges, device=self.device, dtype=torch.long))
            refs, scores = self.retriever.query_batch(z.detach().cpu().numpy(), mask.cpu().numpy(), excluded_centers=centers)
            prompt, _, paths = self.builder.build(z, mask, nodes, int(centers[0]), refs, scores)
            prompts.append(prompt)
            references.append(refs)
            labels.append(torch.as_tensor(np.array(self.bank['y'][rows]), device=self.device, dtype=torch.long))
            reachable += paths
        return (prompts, labels, references, reachable)
