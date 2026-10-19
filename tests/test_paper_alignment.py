import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
import torch
from scrad.model import DualEdgeEncoder
from scrad_complete.anomalies import inject_anomalies
from scrad_complete.frozen_llm import FrozenBinaryLLM
from scrad_complete.protocol import PAPER_REVISION, SETTINGS, corpus_epoch_batches, validate_paper_splits
from scrad_complete.tokenization import GraphTokenizer, SoftPromptBuilder

def tiny_llm():
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
    config._attn_implementation = 'sdpa'
    llm = FrozenBinaryLLM.__new__(FrozenBinaryLLM)
    torch.nn.Module.__init__(llm)
    llm.model = Qwen2Model(config).requires_grad_(False)
    llm.register_buffer('binary_ids', torch.tensor([10, 11]))
    llm.register_buffer('decoder_rows', torch.randn(2, 16))
    llm.hidden_size = 16
    llm.prefix_cache = {}
    llm.offload_activations = False
    return llm

class PaperAlignmentTests(unittest.TestCase):

    def test_frozen_dictionary_linear_projection_and_straight_through(self):
        torch.manual_seed(21)
        tokenizer = GraphTokenizer(np.eye(4, dtype=np.float32), 8)
        before = tokenizer.codebook.clone()
        z = torch.randn(2, 13, 4, requires_grad=True)
        mask = torch.ones(2, 13, dtype=torch.bool)
        tokens, codes, auxiliary = tokenizer(z, mask)
        torch.testing.assert_close(tokens, tokenizer.project(tokenizer.codebook[codes]))
        self.assertEqual(float(auxiliary), 0.0)
        self.assertNotIn('codebook', dict(tokenizer.named_parameters()))
        optimizer = torch.optim.Adam(tokenizer.parameters(), lr=0.01)
        tokens.square().sum().backward()
        optimizer.step()
        self.assertGreater(float(z.grad.norm()), 0.0)
        torch.testing.assert_close(tokenizer.codebook, before, atol=0, rtol=0)

    def test_nll_through_frozen_transformer_updates_edge_encoder_and_projection(self):
        torch.manual_seed(8)
        dual = DualEdgeEncoder(4, depth=2, local_hops=1, paper=True)
        tokens = GraphTokenizer(np.random.default_rng(2).normal(size=(16, 4)), 16)
        llm = tiny_llm()
        h = torch.randn(7, 4)
        scales = torch.randn(6, 2, 4)
        seq = torch.arange(7)[None]
        edges = torch.arange(6)[None]
        z, mask, _ = dual(h, scales, seq, edges)
        prompt, _, _ = tokens(z, mask)
        old_gate = dual.gate.detach().clone()
        old_book = tokens.codebook.clone()
        optimizer = torch.optim.Adam([p for p in dual.parameters() if p.requires_grad] + list(tokens.parameters()), lr=0.01)
        loss, _ = llm.binary_loss([prompt[0]], [torch.tensor([1])])
        loss.backward()
        self.assertGreater(float(dual.gate.grad.norm()), 0.0)
        self.assertGreater(float(tokens.project.weight.grad.norm()), 0.0)
        self.assertFalse(any((p.grad is not None for p in llm.parameters())))
        self.assertIsNone(tokens.codebook.grad)
        optimizer.step()
        self.assertFalse(torch.equal(dual.gate, old_gate))
        torch.testing.assert_close(tokens.codebook, old_book, rtol=0, atol=0)

    def test_variable_view_nll_matches_sequence_weighted_individual_losses(self):
        torch.manual_seed(5)
        llm = tiny_llm()
        prompts = [torch.randn(4, 16, requires_grad=True), torch.randn(6, 16, requires_grad=True)]
        labels = [torch.tensor([1]), torch.tensor([0, 1, 0])]
        loss, logits = llm.binary_loss(prompts, labels)
        expected = sum((llm.binary_loss([p], [y])[0] * len(y) for p, y in zip(prompts, labels))) / 4
        torch.testing.assert_close(loss, expected, atol=1e-06, rtol=1e-06)
        first = torch.autograd.grad(loss, prompts, retain_graph=True)
        second = torch.autograd.grad(expected, prompts)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, atol=1e-05, rtol=1e-05)
        self.assertEqual(logits.shape, (2, 3, 2))

    def test_local_sum_attention_and_gate_have_no_extra_normalization(self):
        model = DualEdgeEncoder(2, depth=1, local_hops=1, paper=True)
        with torch.no_grad():
            model.left_hops[0].weight.copy_(torch.eye(2))
            model.right_hops[0].weight.copy_(torch.eye(2))
            model.direction_score.weight.zero_()
            model.global_encoder.weight.zero_()
            model.gate.zero_()
        nodes = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]])
        elements, _, _ = model(nodes, torch.zeros(2, 1, 2), torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1]]))
        a, b = (nodes[:-1] + nodes[1:]) / 2
        expected = 0.5 * torch.stack([(a + a + b) / 2, (a + b + b) / 2])
        torch.testing.assert_close(elements[0, 1::2], expected)

    def test_corpus_epoch_visits_each_training_sequence_once(self):
        targets = np.repeat([4, 9, 17, 18], [1, 20, 7, 2])
        a = corpus_epoch_batches(targets, 3, 9, np.random.default_rng(42))
        b = corpus_epoch_batches(targets, 3, 9, np.random.default_rng(43))
        for batches in (a, b):
            rows = []
            for batch in batches:
                self.assertLessEqual(len(batch), 3)
                for group in batch:
                    self.assertLessEqual(len(group), 9)
                    self.assertEqual(len(np.unique(targets[group])), 1)
                    rows.extend(group)
            np.testing.assert_array_equal(np.sort(rows), np.arange(len(targets)))
        self.assertNotEqual([r.tolist() for batch in a for r in batch], [r.tolist() for batch in b for r in batch])

    def test_split_validation_rejects_label_leakage_and_native_temporal_ratios(self):
        splits = dict(train=np.arange(4), val=np.arange(4, 6), test=np.arange(6, 10))
        validate_paper_splits(splits, 10, splits['train'])
        with self.assertRaisesRegex(ValueError, 'training targets only'):
            validate_paper_splits(splits, 10, np.r_[splits['train'], 6])
        with self.assertRaisesRegex(ValueError, '40/20/40'):
            validate_paper_splits(dict(train=np.arange(7), val=np.array([7]), test=np.array([8, 9])), 10)

    def test_exact_clique_group_counts_override_rounded_anomaly_percentages(self):
        x = np.random.default_rng(1).normal(size=(150, 3)).astype(np.float32)
        result = inject_anomalies(x, sparse.csr_matrix((150, 150)), clique_size=15, clique_groups=3, candidates=5)
        _, graph, labels, splits, record = result
        self.assertEqual(int(labels.sum()), 90)
        np.testing.assert_array_equal(np.diff(record['clique_offsets']), [15, 15, 15])
        self.assertEqual(len(record['contextual_nodes']), 45)
        self.assertEqual(graph.nnz, 3 * 15 * 14)
        self.assertEqual(2 * 15 * SETTINGS['reddit']['clique_groups'], 11190)
        self.assertEqual(2 * 15 * SETTINGS['questions']['clique_groups'], 3570)
        validate_paper_splits(splits, len(labels), splits['train'])

    def test_legacy_token_state_remains_loadable(self):
        source = GraphTokenizer(np.eye(4, dtype=np.float32), 8, legacy=True)
        restored = GraphTokenizer(np.zeros((4, 4), np.float32), 8, legacy=True)
        restored.load_state_dict(source.state_dict())
        z = torch.randn(1, 13, 4)
        mask = torch.ones(1, 13, dtype=torch.bool)
        torch.testing.assert_close(restored(z, mask)[0], source(z, mask)[0])

    def test_short_sequences_describe_the_actual_target_position(self):
        from scrad_complete.kg import SequenceKG

        class Tokenizer:

            def apply_chat_template(self, *args, **kwargs):
                return 'GRAPH_INPUT_PLACEHOLDER'

        class LLM:
            tokenizer = Tokenizer()
            decoder_rows = torch.zeros(2, 8)

            def __init__(self):
                self.texts = []

            def embed_text(self, value):
                self.texts.append(value)
                return torch.zeros(1, 8)
        nodes = np.array([[-1, -1, -1, 0, 9, -1, -1], [-1, -1, -1, 1, 9, -1, -1]])
        mask = np.zeros((2, 13), bool)
        mask[:, 6:9] = True
        bank = dict(nodes=nodes, target=np.array([0, 1]), y=np.array([0, 1]), z=np.ones((2, 13, 4), np.float32), mask=mask, examples=np.array([0, 1]))
        llm = LLM()
        builder = SoftPromptBuilder(llm, GraphTokenizer(np.eye(4), 8), bank, SequenceKG(nodes, bank['target']))
        builder.build(torch.ones(1, 13, 4), torch.from_numpy(mask[:1]), np.array([[-1, -1, -1, 2, 9, -1, -1]]), 2, np.array([[0, 1]]))
        self.assertIn('Target sequence T0 (target position 1): ', llm.texts)
        self.assertNotIn('the middle node is the target', ''.join(llm.texts))

    def test_unquantized_safetensors_loader_uses_original_weights(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM

        class Tokenizer:

            def encode(self, text, **kwargs):
                return [10 + int(text)]
        torch.manual_seed(1)
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
        original = Qwen2ForCausalLM(config).to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()):
                loaded = FrozenBinaryLLM(directory, device='cpu', gradient_checkpointing=False, offload_activations=False)
            self.assertFalse(loaded.quantized)
            for name, value in original.model.state_dict().items():
                torch.testing.assert_close(loaded.model.state_dict()[name], value, rtol=0, atol=0)
            torch.testing.assert_close(loaded.decoder_rows, original.lm_head.weight[[10, 11]], rtol=0, atol=0)
            prompt = torch.randn(3, 16, requires_grad=True)
            loaded.binary_loss([prompt], [torch.tensor([1])])[0].backward()
            self.assertGreater(float(prompt.grad.norm()), 0.0)
            self.assertFalse(any((p.grad is not None for p in loaded.parameters())))

    def test_cpu_preparation_runtime_label_isolation_and_compact_export(self):
        from scrad_complete.runtime import CompleteRuntime
        from scrad_complete.export_inference import export

        class Tokenizer:

            def apply_chat_template(self, *args, **kwargs):
                return 'USER GRAPH_INPUT_PLACEHOLDER ASSISTANT'

        class PromptLLM:
            tokenizer = Tokenizer()
            hidden_size = 16
            decoder_rows = torch.zeros(2, 16)
            quantized = False

            def embed_text(self, value):
                return torch.zeros(1, 16)
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root / 'src'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = folder / 'data'
            data.mkdir()
            rng = np.random.default_rng(7)
            x = rng.normal(size=(40, 5)).astype(np.float32)
            y = np.arange(40) % 2
            splits = dict(train=np.arange(16), val=np.arange(16, 24), test=np.arange(24, 40))
            adj = sparse.csr_matrix(np.ones((40, 40), np.float32) - np.eye(40, dtype=np.float32))
            sparse.save_npz(data / 'questions_adj.npz', adj)

            def run(module, *args):
                result = subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=root, env=env, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
            initial = []
            for iteration in range(2):
                if iteration:
                    y[16:] = 1 - y[16:]
                np.savez(data / 'questions.npz', x=x, y=y, **splits)
                run('scrad_complete.pretrain', '--dataset', 'questions', '--data', data, '--output', folder / f'init{iteration}', '--device', 'cpu', '--pmp-epochs', 1)
                initial.append(torch.load(folder / f'init{iteration}/checkpoint.pt', weights_only=True))
            for key in initial[0]['pmp_state_dict']:
                torch.testing.assert_close(initial[0]['pmp_state_dict'][key], initial[1]['pmp_state_dict'][key], atol=2e-05, rtol=2e-05)
            run('scrad_complete.prepare', '--checkpoint', folder / 'init1/checkpoint.pt', '--data', data, '--output', folder / 'prepared', '--device', 'cpu', '--corpus-size', 64, '--include-validation')
            prepared = folder / 'prepared'
            self.assertEqual(np.load(prepared / 'codebook.npy').shape, (512, 256))
            np.testing.assert_array_equal(np.load(prepared / 'test/query_node_ids.npy'), splits['test'])
            np.testing.assert_allclose(initial[1]['scaler_mean'].numpy(), x[splits['train']].mean(0), atol=1e-06)
            nodes = np.load(prepared / 'bank/nodes.npy')
            self.assertTrue(np.isin(nodes, np.r_[splits['val'], splits['test']]).any())
            self.assertTrue(np.isin(np.load(prepared / 'bank/target.npy'), splits['train']).all())
            (prepared / 'evaluation.npz').unlink()
            runtime = CompleteRuntime(prepared, PromptLLM(), torch.device('cpu'))
            prompts, _, refs, paths = runtime.prompts([24])
            self.assertEqual(refs.shape, (1, 9, 5))
            self.assertEqual(paths, 45)
            cp = folder / 'checkpoint.pt'
            torch.save(dict(dual=runtime.dual.state_dict(), tokens=runtime.tokens.state_dict(), memory=runtime.retriever.memory.state_dict(), prepared_config=runtime.cfg, config={'prompt_format': 'verbose'}), cp)
            compact = folder / 'compact'
            export(prepared, cp, compact)
            source = CompleteRuntime(prepared, PromptLLM(), torch.device('cpu'), checkpoint=cp)
            exported = CompleteRuntime(compact, PromptLLM(), torch.device('cpu'), checkpoint=compact / 'checkpoint.pt', inference_only=True)
            a, _, ra, pa = source.prompts([24])
            b, _, rb, pb = exported.prompts([24])
            torch.testing.assert_close(a[0], b[0])
            np.testing.assert_array_equal(ra, rb)
            self.assertEqual(pa, pb)
            from transformers import Qwen2Config, Qwen2ForCausalLM
            from scrad_complete.train import main as train_main

            class TrainingTokenizer(Tokenizer):

                def encode(self, value, **kwargs):
                    return [10 + int(value)] if value in ('0', '1') else [2]
            model_dir = folder / 'tiny_llm'
            config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
            Qwen2ForCausalLM(config).to(torch.bfloat16).save_pretrained(model_dir)

            def align(output, steps, resume=None):
                arguments = ['train', '--run', str(prepared), '--output', str(output), '--llm', str(model_dir), '--device', 'cpu', '--epochs', '1', '--max-steps', str(steps), '--batch-nodes', '2', '--micro-batch-nodes', '1', '--save-every', '1', '--keep-activations-on-gpu']
                if resume:
                    arguments += ['--resume-from', str(resume)]
                with patch('sys.argv', arguments), patch('transformers.AutoTokenizer.from_pretrained', return_value=TrainingTokenizer()):
                    train_main()
                return torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
            partial = align(folder / 'partial', 2)
            resumed = align(folder / 'resumed', 3, folder / 'partial/last.pt')
            uninterrupted = align(folder / 'uninterrupted', 3)
            for key in ('dual', 'tokens', 'memory'):
                for name in uninterrupted[key]:
                    torch.testing.assert_close(resumed[key][name], uninterrupted[key][name], atol=0, rtol=0)
            torch.testing.assert_close(partial['tokens']['codebook'], torch.from_numpy(np.load(prepared / 'codebook.npy')), atol=0, rtol=0)
            self.assertEqual(resumed['sequences_seen'], uninterrupted['sequences_seen'])
            self.assertTrue(json.loads((folder / 'resumed/complete.json').read_text())['pilot'])
            targets = np.load(prepared / 'bank/target.npy')
            targets[0] = 24
            np.save(prepared / 'bank/target.npy', targets)
            with self.assertRaisesRegex(ValueError, 'non-training target'):
                CompleteRuntime(prepared, PromptLLM(), torch.device('cpu'))
if __name__ == '__main__':
    unittest.main()
