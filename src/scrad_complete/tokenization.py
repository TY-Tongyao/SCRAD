import torch
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F

class GraphTokenizer(nn.Module):

    def __init__(self, codebook, llm_dim, embedding_scale=0.03):
        super().__init__()
        self.codebook = nn.Parameter(torch.as_tensor(codebook, dtype=torch.float32).clone())
        dim = self.codebook.shape[1]
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, llm_dim))
        self.kind = nn.Embedding(3, llm_dim)
        nn.init.normal_(self.kind.weight, std=0.01)
        self.log_scale = nn.Parameter(torch.tensor(float(embedding_scale)).log())

    def codes(self, z):
        flat = z.reshape(-1, z.shape[-1])
        distance = flat.square().sum(-1, keepdim=True) + self.codebook.square().sum(-1) - 2 * flat @ self.codebook.T
        return distance.argmin(-1).reshape(z.shape[:-1])

    def forward(self, z, mask):
        codes = self.codes(z)
        quantized = F.embedding(codes, self.codebook)
        vq_loss = (quantized[mask] - z[mask].detach()).square().mean()
        vq_loss = vq_loss + 0.25 * (z[mask] - quantized[mask].detach()).square().mean()
        straight_through = z + (quantized - z).detach()
        projected = self.project(straight_through)
        projected = F.layer_norm(projected, (projected.shape[-1],))
        kind = torch.arange(z.shape[-2], device=z.device) % 2
        kind[z.shape[-2] // 2] = 2
        scale = self.log_scale.exp().clamp(0.005, 0.3)
        tokens = scale * projected + self.kind(kind)
        return (tokens, codes, vq_loss)

class SoftPromptBuilder:

    def __init__(self, llm, graph_tokenizer, bank, kg, prompt_format='verbose'):
        self.llm, self.graph_tokenizer, self.bank, self.kg = (llm, graph_tokenizer, bank, kg)
        if prompt_format not in ['verbose', 'shared_paths']:
            raise ValueError('Unknown prompt format')
        self.prompt_format = prompt_format
        self.cache = OrderedDict()
        template = llm.tokenizer.apply_chat_template([{'role': 'user', 'content': 'GRAPH_INPUT_PLACEHOLDER'}], tokenize=False, add_generation_prompt=True)
        self.chat_prefix, self.chat_suffix = template.split('GRAPH_INPUT_PLACEHOLDER')
        if '<think>' in self.chat_suffix:
            self.chat_suffix += '\n</think>\n'
        import numpy as np
        self.examples = bank['examples'].tolist() if 'examples' in bank else [int(np.flatnonzero(bank['y'] == label)[0]) for label in (0, 1)]

    def text(self, value):
        if value not in self.cache:
            self.cache[value] = self.llm.embed_text(value).detach()
            if len(self.cache) > 512:
                self.cache.popitem(last=False)
        self.cache.move_to_end(value)
        return self.cache[value]

    def build(self, target_z, target_mask, target_nodes, center, refs, weighted_scores=None):
        import numpy as np
        device = target_z.device
        unique = list(dict.fromkeys(self.examples + refs.ravel().tolist()))
        reference_z = torch.as_tensor(self.bank['z'][unique], dtype=torch.float32, device=device)
        reference_mask = torch.as_tensor(self.bank['mask'][unique], device=device)
        ref_tokens, _, ref_loss = self.graph_tokenizer(reference_z, reference_mask)
        target_tokens, _, target_loss = self.graph_tokenizer(target_z, target_mask)
        index = {ref: position for position, ref in enumerate(unique)}
        parts = [self.text(self.chat_prefix), self.text('You classify graph-sequence coherence. Graph tokens alternate nodes and edges; the middle node is the target. Output 0 for coherent, 1 for incoherent. Use labeled examples, retrieved references and shared-attribute KG paths. Do not infer a label from an entity identifier.\n')]
        for label, reference in enumerate(self.examples):
            j = index[reference]
            parts += [self.text(f'Example {label}, reference R{j}: '), ref_tokens[j, reference_mask[j]], self.text(f" Label {int(self.bank['y'][reference])}.\n")]
        for reference in unique:
            if reference in self.examples:
                continue
            j = index[reference]
            parts += [self.text(f'Reference R{j}: '), ref_tokens[j, reference_mask[j]], self.text(f" Label {int(self.bank['y'][reference])}.\n")]
        self.last_prefix = (tuple(unique), sum((len(part) for part in parts)))
        reachable = 0
        path_catalog = {}
        for i in range(len(target_z)):
            parts += [self.text(f'Target sequence T{i}: '), target_tokens[i, target_mask[i]]]
            parts.append(self.text(' References: ' + ','.join((f'R{index[int(r)]}' for r in refs[i])) + '.\n'))
            paths = self.kg.shortest_paths(target_nodes[i], int(center), self.bank['target'][refs[i]])
            path_links = []
            for reference in refs[i]:
                path = paths[int(self.bank['target'][reference])]
                reachable += int(path['reachable'])
                description = self.kg.verbalize(path)
                if self.prompt_format == 'verbose':
                    parts.append(self.llm.embed_text(f'T{i} to R{index[int(reference)]}: {description}\n').detach())
                else:
                    path_id = path_catalog.setdefault(description, len(path_catalog))
                    path_links.append(f'R{index[int(reference)]}=P{path_id}')
            if path_links:
                parts.append(self.text(' KG paths: ' + ','.join(path_links) + '.\n'))
        if path_catalog:
            description = 'Shared KG path descriptions (each target-reference link above uses its P identifier):\n'
            description += '\n'.join((f'P{path_id}: {path}' for path, path_id in path_catalog.items())) + '\n'
            parts.append(self.llm.embed_text(description).detach())
        parts.append(self.text(f'Give exactly {len(target_z)} binary decisions in target sequence order, with no explanation.'))
        parts.append(self.text(self.chat_suffix + 'Decisions: '))
        return (torch.cat([part.to(self.llm.decoder_rows.dtype) for part in parts], 0), (target_loss + ref_loss) / 2, reachable)
