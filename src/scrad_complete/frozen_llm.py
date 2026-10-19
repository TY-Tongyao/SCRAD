import json
import atexit
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import traceback
import torch
from torch import nn
from torch.nn import functional as F

def dequantize_reference(qweight, qzeros, scales):
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=qweight.device)
    shifts = order * 4
    weights = (qweight.unsqueeze(-1) >> shifts & 15).reshape(qweight.shape[0], -1)
    zeros = (qzeros.unsqueeze(-1) >> shifts & 15).reshape(qzeros.shape[0], -1)
    group = qweight.shape[0] // scales.shape[0]
    return (weights.to(scales.dtype) - zeros.repeat_interleave(group, 0)) * scales.repeat_interleave(group, 0)

def dequantize(qweight, qzeros, scales):
    if qweight.is_cuda:
        from vllm import _custom_ops as ops
        return ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
    return dequantize_reference(qweight, qzeros, scales)

class _FrozenAWQ(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, qweight, qzeros, scales, bias):
        ctx.quantized_weights = (qweight, qzeros, scales)
        ctx.input_dtype = x.dtype
        qweight, qzeros, scales = [value.to(x.device, non_blocking=True) for value in ctx.quantized_weights]
        if x.is_cuda and x.numel() // x.shape[-1] < 256:
            from vllm import _custom_ops as ops
            result = ops.awq_gemm(x.reshape(-1, x.shape[-1]).to(scales.dtype), qweight, scales, qzeros, 8)
            result = result.reshape(*x.shape[:-1], -1)
        else:
            weight = dequantize(qweight, qzeros, scales)
            result = x.to(weight.dtype) @ weight
        if bias is not None:
            result = result + bias.to(x.device, non_blocking=True)
        return result.to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        qweight, qzeros, scales = [value.to(grad.device, non_blocking=True) for value in ctx.quantized_weights]
        weight = dequantize(qweight, qzeros, scales)
        dx = (grad.to(weight.dtype) @ weight.T).to(ctx.input_dtype)
        return (dx, None, None, None, None)

class FrozenAWQLinear(nn.Module):

    def __init__(self, qweight, qzeros, scales, bias=None):
        super().__init__()
        self.register_buffer('qweight', qweight)
        self.register_buffer('qzeros', qzeros)
        self.register_buffer('scales', scales)
        self.register_buffer('bias', bias)

    def forward(self, x):
        return _FrozenAWQ.apply(x, self.qweight, self.qzeros, self.scales, self.bias)

def _prediction_worker(connection, directory, device, memory_fraction, cpu_weight_layers):
    try:
        torch.set_num_threads(4)
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
        model = FrozenBinaryLLM(directory, device=device, gradient_checkpointing=False, cpu_weight_layers=cpu_weight_layers)
        connection.send({'ready': True})
        while True:
            request = connection.recv()
            if request is None:
                break
            prompts, count, prefixes = request
            prompts = [torch.from_numpy(value).to(device) for value in prompts]
            decisions, probabilities = model.predict(prompts, count=count, prefixes=prefixes)
            connection.send({'decisions': decisions.cpu().tolist(), 'probabilities': probabilities.cpu().tolist(), 'prefix_cache_hits': model.prefix_cache_hits, 'prefix_cache_misses': model.prefix_cache_misses})
            del prompts, request, decisions, probabilities
    except BaseException:
        try:
            connection.send({'error': traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()

class FrozenBinaryWorker:
    uses_cpu_input = True

    def __init__(self, directory, device, memory_fraction, cpu_weight_layers=0):
        import torch.multiprocessing as multiprocessing
        device = torch.device(device)
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_prediction_worker, args=(child, str(directory), device, memory_fraction, cpu_weight_layers), daemon=True)
        self.decoder_rows = SimpleNamespace(device=device)
        self.prefix_cache_hits = self.prefix_cache_misses = 0
        self.process.start()
        child.close()
        atexit.register(self.close)
        self._receive()

    def _receive(self):
        while not self.connection.poll(1):
            if not self.process.is_alive():
                raise RuntimeError(f'Frozen LLM worker exited with code {self.process.exitcode}')
        try:
            result = self.connection.recv()
        except EOFError as error:
            raise RuntimeError('Frozen LLM worker closed without a result') from error
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def predict(self, prompts, count=9, prefixes=None):
        arrays = [prompt.detach().cpu().numpy() for prompt in prompts]
        self.connection.send((arrays, count, prefixes))
        result = self._receive()
        self.prefix_cache_hits = result['prefix_cache_hits']
        self.prefix_cache_misses = result['prefix_cache_misses']
        return (torch.tensor(result['decisions'], dtype=torch.long), torch.tensor(result['probabilities'], dtype=torch.float32))

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
        self.connection.close()

class FrozenBinaryLLM(nn.Module):

    def __init__(self, directory, device='cuda', gradient_checkpointing=True, offload_activations=True, embedding_device='cpu', cpu_weight_layers=0):
        super().__init__()
        from safetensors import safe_open
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
        directory = Path(directory)
        cfg = AutoConfig.from_pretrained(directory, local_files_only=True)
        if cfg.model_type != 'qwen2':
            raise ValueError('This loader supports Qwen2-family AWQ GEMM models')
        quant = cfg.quantization_config
        if quant['bits'] != 4 or quant['version'].lower() != 'gemm':
            raise ValueError('Expected 4-bit AWQ GEMM checkpoint')
        cfg._attn_implementation = 'sdpa'
        if not 0 <= cpu_weight_layers <= cfg.num_hidden_layers:
            raise ValueError('CPU weight layer count exceeds model depth')
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        ids = [self.tokenizer.encode(str(label), add_special_tokens=False) for label in (0, 1)]
        if any((len(x) != 1 for x in ids)):
            raise ValueError('Binary labels must each be exactly one tokenizer token')
        self.register_buffer('binary_ids', torch.tensor([x[0] for x in ids], device=device))
        index = json.loads((directory / 'model.safetensors.index.json').read_text())['weight_map']

        def tensor(key, target_device=device):
            with safe_open(directory / index[key], framework='pt', device='cpu') as f:
                t = f.get_tensor(key)
            return t.to(device=target_device, dtype=torch.float16 if t.is_floating_point() else t.dtype)
        with torch.device('meta'):
            self.model = Qwen2Model(cfg)
        for name, module in list(self.model.named_modules()):
            key = 'model.' + name
            if isinstance(module, nn.Linear) and key + '.qweight' in index:
                parent_name, _, child = name.rpartition('.')
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                layer = int(name.split('.')[1])
                target = 'cpu' if layer >= cfg.num_hidden_layers - cpu_weight_layers else device
                replacement = FrozenAWQLinear(tensor(key + '.qweight', target), tensor(key + '.qzeros', target), tensor(key + '.scales', target), tensor(key + '.bias', target) if key + '.bias' in index else None)
                setattr(parent, child, replacement)
        for name, _ in list(self.model.named_parameters()):
            parent_name, _, child = name.rpartition('.')
            parent = self.model.get_submodule(parent_name) if parent_name else self.model
            target = embedding_device if name == 'embed_tokens.weight' else device
            setattr(parent, child, nn.Parameter(tensor('model.' + name, target), requires_grad=False))
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        self.model.rotary_emb = Qwen2RotaryEmbedding(cfg, device=device)
        with safe_open(directory / index['lm_head.weight'], framework='pt', device='cpu') as f:
            rows = torch.stack([f.get_slice('lm_head.weight')[ident[0], :] for ident in ids])
        self.register_buffer('decoder_rows', rows.to(device=device, dtype=torch.float16))
        self.model.requires_grad_(False)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model.eval()
        self.hidden_size = cfg.hidden_size
        self.directory = str(directory)
        self.prefix_cache = OrderedDict()
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0
        self.offload_activations = offload_activations

    def embed_text(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return self.embed_ids(torch.tensor(ids, dtype=torch.long))

    def embed_ids(self, ids):
        rows = self.model.embed_tokens(ids.to(self.model.embed_tokens.weight.device))
        return rows.to(self.decoder_rows.device)

    def binary_loss(self, prompts, labels):
        self.model.train()
        self.prefix_cache.clear()
        b, count = labels.shape
        lengths = [len(x) for x in prompts]
        total = max(lengths) + count - 1
        embeds = self.decoder_rows.new_zeros((b, total, self.hidden_size))
        attention = torch.zeros((b, total), device=embeds.device, dtype=torch.long)
        read = []
        for i, prompt in enumerate(prompts):
            suffix = self.embed_ids(self.binary_ids[labels[i, :-1]])
            joined = torch.cat([prompt.to(embeds.dtype), suffix], 0)
            embeds[i, :len(joined)] = joined
            attention[i, :len(joined)] = 1
            read.append(torch.arange(len(prompt) - 1, len(prompt) + count - 1, device=embeds.device))
        context = torch.autograd.graph.save_on_cpu(pin_memory=True) if self.offload_activations else nullcontext()
        with context:
            hidden = self.model(inputs_embeds=embeds, attention_mask=attention, use_cache=False).last_hidden_state
            selected = torch.stack([hidden[i, positions] for i, positions in enumerate(read)])
            logits = F.linear(selected, self.decoder_rows).float()
            loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))
        return (loss, logits)

    @torch.no_grad()
    def predict(self, prompts, count=9, prefixes=None):
        self.model.eval()
        past = None
        prefix_lengths = None
        if prefixes is not None:
            from transformers.cache_utils import DynamicCache
            cached = []
            for prompt, (key, length) in zip(prompts, prefixes):
                key = (key, length)
                if key not in self.prefix_cache:
                    output = self.model(inputs_embeds=prompt[None, :length], use_cache=True)
                    self.prefix_cache[key] = tuple(((k.detach(), v.detach()) for k, v in output.past_key_values.to_legacy_cache()))
                    self.prefix_cache_misses += 1
                    if len(self.prefix_cache) > 16:
                        self.prefix_cache.popitem(last=False)
                else:
                    self.prefix_cache_hits += 1
                self.prefix_cache.move_to_end(key)
                cached.append(self.prefix_cache[key])
            prefix_lengths = [item[1] for item in prefixes]
            width = max(prefix_lengths)
            batched = []
            for layer in range(len(cached[0])):
                batched.append(tuple((torch.cat([F.pad(item[layer][kind], (0, 0, width - length, 0)) for item, length in zip(cached, prefix_lengths)], 0) for kind in [0, 1])))
            past = DynamicCache.from_legacy_cache(tuple(batched))
            prompts = [prompt[length:] for prompt, length in zip(prompts, prefix_lengths)]
        b = len(prompts)
        longest = max((len(p) for p in prompts))
        embeds = self.decoder_rows.new_zeros((b, longest, self.hidden_size))
        attention = torch.zeros((b, longest), device=embeds.device, dtype=torch.long)
        for i, prompt in enumerate(prompts):
            embeds[i, -len(prompt):] = prompt.to(embeds.dtype)
            attention[i, -len(prompt):] = 1
        if prefix_lengths is not None:
            prefix_attention = torch.zeros((b, max(prefix_lengths)), device=embeds.device, dtype=torch.long)
            for i, length in enumerate(prefix_lengths):
                prefix_attention[i, -length:] = 1
            attention = torch.cat([prefix_attention, attention], -1)
        decisions, probabilities = ([], [])
        for step in range(count):
            position = attention.long().cumsum(-1) - 1
            position.masked_fill_(attention == 0, 0)
            position = position[:, -embeds.shape[1]:]
            output = self.model(inputs_embeds=embeds, attention_mask=attention, position_ids=position, past_key_values=past, use_cache=True)
            logits = F.linear(output.last_hidden_state[:, -1], self.decoder_rows).float()
            choice = logits.argmax(-1)
            decisions.append(choice)
            probabilities.append(logits.softmax(-1)[:, 1])
            past = output.past_key_values
            embeds = self.embed_ids(self.binary_ids[choice])[:, None]
            attention = torch.cat([attention, torch.ones((b, 1), device=attention.device, dtype=attention.dtype)], -1)
        return (torch.stack(decisions, 1), torch.stack(probabilities, 1))
