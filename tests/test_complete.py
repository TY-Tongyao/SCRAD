import unittest
import numpy as np
import torch
from scrad_complete.kg import SequenceKG
from scrad_complete.memory import AdaptiveMemory, EntropyRetriever
from scrad_complete.frozen_llm import FrozenAWQLinear, FrozenBinaryLLM, dequantize_reference
from scrad_complete.tokenization import GraphTokenizer, SoftPromptBuilder
from scrad_complete.anomalies import inject_anomalies, relocate_contextual_nodes
from scipy import sparse
from scrad.data import Partition, line_graph_scales
from scrad_complete.scalable import fast_sample_sequences, selected_line_graph_scales, pmp_unknown, csr_tensor, original_partition_adjacency, MappedEdgeScales
from scrad.model import PMPEncoder

class CompleteTests(unittest.TestCase):

    def test_inference_resume_changes_only_replica_resources(self):
        import json
        import tempfile
        from pathlib import Path
        from scrad_complete.infer import record_inference_config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            config = {'checkpoint_sha256': 'fixed', 'run': 'original', 'max_batch_tokens': 12000, 'batch_nodes': 32, 'llm_devices': '0,1', 'gpu_memory_fraction': 0.18}
            record_inference_config(path, config)
            torch.save({'completed_nodes': 64}, path.parent / 'memory.pt')
            expanded = config | {'llm_devices': '0,1,2'}
            record_inference_config(path, expanded)
            self.assertEqual(json.loads(path.read_text()), expanded)
            history = json.loads((path.parent / 'resource_changes.json').read_text())
            self.assertEqual(history[0]['completed_nodes'], 64)
            expanded = expanded | {'replica_processes': True}
            record_inference_config(path, expanded)
            self.assertEqual(json.loads(path.read_text()), expanded)
            for key, value in [('checkpoint_sha256', 'other'), ('run', 'different'), ('max_batch_tokens', 18000), ('batch_nodes', 64)]:
                with self.assertRaisesRegex(ValueError, 'Cannot mix'):
                    record_inference_config(path, expanded | {key: value})
            self.assertEqual(json.loads(path.read_text()), expanded)

    def test_compact_deployment_rejects_training_and_mismatched_checkpoint(self):
        import json
        import tempfile
        from pathlib import Path
        from scrad_complete.runtime import CompleteRuntime
        from scrad_complete.anomalies import sha256_stream
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'prepared.json').write_text('{}')
            checkpoint = root / 'checkpoint.pt'
            checkpoint.write_bytes(b'original')
            manifest = {'checkpoint_sha256': sha256_stream(checkpoint), 'files_sha256': {}}
            (root / 'deployment.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'supports inference'):
                CompleteRuntime(root, None, 'cpu', checkpoint=checkpoint)
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'different checkpoint'):
                CompleteRuntime(root, None, 'cpu', checkpoint=checkpoint, inference_only=True)
            checkpoint.write_bytes(b'original')
            manifest['files_sha256'] = {'../outside.npy': 'invalid'}
            (root / 'deployment.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'file mismatch'):
                CompleteRuntime(root, None, 'cpu', checkpoint=checkpoint, inference_only=True)

    def test_reference_edge_cache_preserves_values_and_skips_cached_disk_reads(self):

        class TrackedArray:

            def __init__(self, values):
                self.values = values
                self.shape = values.shape
                self.dtype = values.dtype
                self.reads = []

            def __len__(self):
                return len(self.values)

            def __getitem__(self, rows):
                self.reads.extend(np.asarray(rows).ravel().tolist())
                return self.values[rows]
        values = np.arange(60, dtype=np.float32).reshape(10, 2, 3)
        tracked = TrackedArray(values)
        mapped = MappedEdgeScales(tracked, 'cpu')
        self.assertEqual(mapped.prefetch([7, -1, 2, 7], max_bytes=1), 0)
        self.assertEqual(mapped.prefetch([7, -1, 2, 7], chunk_size=2), 3 * 2 * 3 * 4)
        self.assertEqual(tracked.reads, [0, 2, 7])
        tracked.reads.clear()
        ids = torch.tensor([[7, 2], [0, 7]])
        torch.testing.assert_close(mapped[ids], torch.from_numpy(values[ids.numpy()]), atol=0, rtol=0)
        self.assertEqual(tracked.reads, [])
        ids = torch.tensor([[7, 8], [0, 3]])
        torch.testing.assert_close(mapped[ids], torch.from_numpy(values[ids.numpy()]), atol=0, rtol=0)
        self.assertEqual(tracked.reads, [8, 3])

    def test_context_relocation_keeps_partitions_counts_and_original_features(self):
        rng = np.random.default_rng(22)
        x = rng.normal(size=(18, 4)).astype(np.float32)
        copy = x.copy()
        splits = {name: np.arange(i * 6, (i + 1) * 6) for i, name in enumerate(['train', 'val', 'test'])}
        graph = np.zeros((18, 18), np.float32)
        for ids in splits.values():
            graph[np.ix_(ids[:3], ids[:3])] = 1
        np.fill_diagonal(graph, 0)
        record = {'structural_nodes': np.array([2, 8, 14]), 'clique_offsets': np.array([0, 3]), 'contextual_nodes': np.array([3, 9, 15, 0]), 'contextual_donors': np.array([5, 11, 17, 4])}
        features, y, new = relocate_contextual_nodes(x, sparse.csr_matrix(graph), splits, record, candidates=5)
        np.testing.assert_array_equal(x, copy)
        np.testing.assert_array_equal(record['contextual_nodes'], [3, 9, 15, 0])
        np.testing.assert_array_equal(new['structural_nodes'], record['structural_nodes'])
        self.assertTrue(np.all(new['contextual_source_partition_degree'] >= 2))
        self.assertEqual(new['contextual_nodes'][-1], 0)
        self.assertEqual(new['contextual_donors'][-1], 4)
        old_y = np.zeros(18, np.int64)
        old_y[np.r_[record['structural_nodes'], record['contextual_nodes']]] = 1
        for ids in splits.values():
            self.assertEqual(int(old_y[ids].sum()), int(y[ids].sum()))
        np.testing.assert_array_equal(features[new['contextual_nodes']], x[new['contextual_donors']])
        keep = np.ones(18, bool)
        keep[new['contextual_nodes']] = False
        np.testing.assert_array_equal(features[keep], x[keep])
        second = relocate_contextual_nodes(x, sparse.csr_matrix(graph), splits, record, candidates=5)
        np.testing.assert_array_equal(second[0], features)
        with self.assertRaisesRegex(ValueError, 'Not enough eligible'):
            relocate_contextual_nodes(x, sparse.csr_matrix(graph), splits, record, minimum_degree=3)

    def test_token_budget_batches_preserve_real_transformer_decisions(self):
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
        from scrad_complete.infer import predict_in_token_budget
        torch.manual_seed(31)
        cfg = Qwen2Config(vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
        cfg._attn_implementation = 'sdpa'
        llm = FrozenBinaryLLM.__new__(FrozenBinaryLLM)
        torch.nn.Module.__init__(llm)
        llm.model = Qwen2Model(cfg)
        llm.model.requires_grad_(False)
        llm.register_buffer('binary_ids', torch.tensor([10, 11]))
        llm.register_buffer('decoder_rows', torch.randn(2, 16))
        llm.hidden_size = 16
        prompts = [torch.randn(length, 16) for length in [3, 7, 2, 11]]
        expected = llm.predict(prompts, count=3)
        actual = predict_in_token_budget(llm, prompts, count=3, max_tokens=18)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1], atol=1e-06, rtol=1e-06)
        import copy
        replica = copy.deepcopy(llm)
        parallel = predict_in_token_budget(llm, prompts, count=3, max_tokens=18, replicas=[llm, replica])
        torch.testing.assert_close(parallel[0], expected[0])
        torch.testing.assert_close(parallel[1], expected[1], atol=1e-06, rtol=1e-06)
        three = predict_in_token_budget(llm, prompts, count=3, max_tokens=18, replicas=[llm, replica, copy.deepcopy(llm)])
        torch.testing.assert_close(three[0], expected[0])
        torch.testing.assert_close(three[1], expected[1], atol=1e-06, rtol=1e-06)
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            predict_in_token_budget(llm, prompts, count=3, max_tokens=4)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for CPU-to-GPU streaming equivalence')
    def test_host_streamed_awq_weights_preserve_forward_and_gradient(self):
        torch.manual_seed(23)
        q = torch.randint(-2 ** 30, 2 ** 30, (128, 16), dtype=torch.int32)
        zeros = torch.randint(-2 ** 30, 2 ** 30, (1, 16), dtype=torch.int32)
        scales = torch.rand(1, 128, dtype=torch.float16) * 0.01
        bias = torch.randn(128, dtype=torch.float16)
        host = FrozenAWQLinear(q, zeros, scales, bias)
        resident = FrozenAWQLinear(q.cuda(), zeros.cuda(), scales.cuda(), bias.cuda())
        for batch in [2, 300]:
            x = torch.randn(batch, 128, device='cuda', dtype=torch.float16, requires_grad=True)
            a, b = (host(x), resident(x))
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            ga = torch.autograd.grad(a.float().square().mean(), x)[0]
            gb = torch.autograd.grad(b.float().square().mean(), x)[0]
            torch.testing.assert_close(ga, gb, atol=0, rtol=0)
        self.assertFalse(host.qweight.is_cuda)

    def test_original_partitions_preserve_forward_gradient_and_sampling(self):
        torch.manual_seed(19)
        rng = np.random.default_rng(19)
        source = sparse.csr_matrix(np.triu(rng.random((15, 15)) < 0.5, 1), dtype=np.float32)
        source = source + source.T
        splits = {'train': np.array([0, 3, 6, 9, 12, 14]), 'val': np.array([1, 4, 7]), 'test': np.array([2, 5, 8, 10, 11, 13])}
        blocked = original_partition_adjacency(source, splits)
        membership = np.empty(15, np.int32)
        for group, ids in enumerate(splits.values()):
            membership[ids] = group
        rows, cols = blocked.nonzero()
        self.assertTrue(np.all(membership[rows] == membership[cols]))
        model = PMPEncoder(3, 4)
        x = torch.randn(15, 3, requires_grad=True)
        full, _ = pmp_unknown(model, x, csr_tensor(blocked, 'cpu'))
        for ids in splits.values():
            local, _ = pmp_unknown(model, x[ids], csr_tensor(source[ids][:, ids], 'cpu'))
            torch.testing.assert_close(full[ids], local)
            expected = torch.autograd.grad(local.square().sum(), x, retain_graph=True)[0]
            actual = torch.autograd.grad(full[ids].square().sum(), x, retain_graph=True)[0]
            torch.testing.assert_close(actual, expected)
        u, v = sparse.triu(blocked, k=1).nonzero()
        numbers = np.arange(len(u)) + 1
        lookup = sparse.csr_matrix((np.r_[numbers, numbers], (np.r_[u, v], np.r_[v, u])), shape=blocked.shape)
        part = Partition(np.arange(15), None, None, blocked, np.stack([u, v], 1), lookup)
        centers = np.repeat(np.arange(15), 9)
        sequences, _ = fast_sample_sequences(part, centers, rng)
        for center, sequence in zip(centers, sequences):
            self.assertTrue(np.all(membership[sequence[sequence >= 0]] == membership[center]))

    def test_offloaded_checkpointed_transformer_preserves_input_gradient(self):
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
        torch.manual_seed(7)
        cfg = Qwen2Config(vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
        cfg._attn_implementation = 'sdpa'
        llm = FrozenBinaryLLM.__new__(FrozenBinaryLLM)
        torch.nn.Module.__init__(llm)
        llm.model = Qwen2Model(cfg)
        for name, module in list(llm.model.named_modules()):
            if isinstance(module, torch.nn.Linear):
                parent, _, child = name.rpartition('.')
                q = torch.randint(-2 ** 30, 2 ** 30, (module.in_features, module.out_features // 8), dtype=torch.int32)
                zeros = torch.randint(-2 ** 30, 2 ** 30, (module.in_features // 4, module.out_features // 8), dtype=torch.int32)
                scales = torch.full((module.in_features // 4, module.out_features), 0.02)
                setattr(llm.model.get_submodule(parent), child, FrozenAWQLinear(q, zeros, scales))
        llm.model.requires_grad_(False)
        llm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        llm.register_buffer('binary_ids', torch.tensor([10, 11]))
        llm.register_buffer('decoder_rows', torch.randn(2, 16))
        llm.hidden_size = 16
        llm.prefix_cache = {}
        values = [torch.randn(length, 16) for length in [5, 7]]
        labels = torch.tensor([[0, 0, 0], [1, 1, 1]])
        results = []
        for offload in [False, True]:
            llm.offload_activations = offload
            prompts = [value.clone().requires_grad_() for value in values]
            loss, _ = llm.binary_loss(prompts, labels)
            loss.backward()
            results.append((loss.detach(), [prompt.grad for prompt in prompts]))
        torch.testing.assert_close(results[0][0], results[1][0])
        for a, b in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(a, b)
            self.assertGreater(float(a.norm()), 0.0)
        self.assertFalse(any((parameter.grad is not None for parameter in llm.parameters())))

    def test_shared_path_prompt_preserves_every_target_reference_link(self):

        class FakeTokenizer:

            def apply_chat_template(self, *args, **kwargs):
                return 'USER:GRAPH_INPUT_PLACEHOLDER ASSISTANT:'

        class FakeLLM:
            tokenizer = FakeTokenizer()
            decoder_rows = torch.zeros(2, 8)

            def __init__(self):
                self.texts = []

            def embed_text(self, text):
                self.texts.append(text)
                return torch.zeros(len(text), 8)
        rng = np.random.default_rng(4)
        nodes = np.array([[10, 1, 20], [20, 2, 30]])
        bank = {'z': rng.normal(size=(2, 13, 4)).astype(np.float32), 'mask': np.ones((2, 13), bool), 'nodes': nodes, 'target': np.array([1, 2]), 'y': np.array([0, 1]), 'examples': np.array([0, 1])}
        kg = SequenceKG(nodes, bank['target'])
        target = torch.as_tensor(rng.normal(size=(9, 13, 4)).astype(np.float32))
        mask = torch.ones((9, 13), dtype=torch.bool)
        targets = np.tile([10, 100, 90], (9, 1))
        refs = np.tile([0, 1], (9, 1))
        tokenizer = GraphTokenizer(np.eye(4, dtype=np.float32), 8)
        verbose, shared = (FakeLLM(), FakeLLM())
        a = SoftPromptBuilder(verbose, tokenizer, bank, kg, 'verbose').build(target, mask, targets, 100, refs)
        b = SoftPromptBuilder(shared, tokenizer, bank, kg, 'shared_paths').build(target, mask, targets, 100, refs)
        self.assertEqual(a[2], b[2])
        self.assertEqual(b[2], 18)
        self.assertLess(len(b[0]), len(a[0]))
        catalog = next((text for text in shared.texts if text.startswith('Shared KG path descriptions')))
        self.assertEqual(catalog.count('P0:'), 1)
        self.assertEqual(catalog.count('P1:'), 1)
        self.assertIn(' KG paths: R0=P0,R1=P1.\n', shared.texts)
        for path in kg.shortest_paths(targets[0], 100, [1, 2]).values():
            self.assertIn(kg.verbalize(path), catalog)

    def test_unknown_pmp_matches_general_forward_and_gradients(self):
        torch.manual_seed(5)
        model = PMPEncoder(3, 4)
        adj = csr_tensor(sparse.csr_matrix(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], np.float32)), 'cpu')
        x = torch.randn(3, 3, requires_grad=True)
        expected, _ = model(x, adj, torch.full((3,), 2))
        actual, _ = pmp_unknown(model, x, adj)
        torch.testing.assert_close(actual, expected)
        first = torch.autograd.grad(expected.square().sum(), x, retain_graph=True)[0]
        second = torch.autograd.grad(actual.square().sum(), x)[0]
        torch.testing.assert_close(first, second)

    def test_selected_line_graph_matches_full_propagation(self):
        rng = np.random.default_rng(31)
        adj = sparse.csr_matrix(np.triu(rng.random((18, 18)) < 0.2, 1), dtype=np.float32)
        adj = adj + adj.T
        u, v = sparse.triu(adj, k=1).nonzero()
        edge_ids = np.arange(len(u)) + 1
        lookup = sparse.csr_matrix((np.r_[edge_ids, edge_ids], (np.r_[u, v], np.r_[v, u])), shape=adj.shape)
        part = Partition(np.arange(18), None, None, adj, np.stack([u, v], 1), lookup)
        h = rng.normal(size=(18, 5)).astype(np.float32)
        selected = np.array([len(u) - 1, 0, 2, 2])
        for restart in [0.0, 0.1, 1.0]:
            full = line_graph_scales(part, h, depth=4, restart=restart)
            actual = selected_line_graph_scales(part, h, selected, depth=4, restart=restart, chunk_size=2)
            np.testing.assert_allclose(actual, full[selected], atol=4e-06, rtol=4e-06)
        centers = np.repeat(np.arange(18), 7)
        seq, edges = fast_sample_sequences(part, centers, np.random.default_rng(5))
        np.testing.assert_array_equal(seq[:, 3], centers)
        for row, eid in zip(seq, edges):
            nodes = row[row >= 0]
            self.assertEqual(len(set(nodes)), len(nodes))
            self.assertTrue(np.all(eid[(row[:-1] >= 0) & (row[1:] >= 0)] >= 0))

    def test_cached_kg_paths_match_augmented_graph_shortest_lengths(self):
        from scipy.sparse.csgraph import shortest_path
        rng = np.random.default_rng(14)
        centers = np.repeat(np.arange(15), 3)
        seq = rng.integers(15, 35, (len(centers), 5))
        seq[:, 2] = centers
        kg = SequenceKG(seq, centers)
        for center in [0, 100]:
            query = np.array([16, 31, center, 90, 91])
            paths = kg.shortest_paths(query, center, range(15))
            attributes = set(query[query != center])
            if center in kg.lookup:
                attributes.update(kg.incidence.getrow(kg.lookup[center]).indices)
            sets = [set(kg.incidence.getrow(i).indices) for i in range(15)] + [attributes]
            entity_adj = np.array([[bool(a & b) for b in sets] for a in sets], np.int32)
            np.fill_diagonal(entity_adj, 0)
            if center == 0:
                entity_adj[15, 0] = entity_adj[0, 15] = 0
            distance = shortest_path(sparse.csr_matrix(entity_adj), directed=False, unweighted=True, indices=15)
            for ref, path in paths.items():
                if ref == center:
                    continue
                self.assertEqual(len(path['attributes']), int(distance[ref]))
                for a, attribute, b in zip(path['entities'][:-1], path['attributes'], path['entities'][1:]):
                    self.assertIn(attribute, attributes if a == center else sets[a])
                    self.assertIn(attribute, sets[b])

    def test_anomaly_generation_preserves_source_and_is_reproducible(self):
        x = np.random.default_rng(5).normal(size=(100, 4)).astype(np.float32)
        adj = sparse.csr_matrix((100, 100), dtype=np.float32)
        original = x.copy()
        result = inject_anomalies(x, adj, ratio=0.22, clique_size=5, candidates=8)
        modified, graph, y, splits, record = result
        np.testing.assert_array_equal(x, original)
        self.assertEqual(adj.nnz, 0)
        self.assertEqual(int(y.sum()), 22)
        np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(100))
        self.assertEqual([len(splits[key]) for key in ['train', 'val', 'test']], [40, 20, 40])
        self.assertEqual((graph != graph.T).nnz, 0)
        self.assertFalse(graph.diagonal().any())
        keep = np.ones(100, bool)
        keep[record['contextual_nodes']] = False
        np.testing.assert_array_equal(modified[keep], x[keep])
        np.testing.assert_array_equal(modified[record['contextual_nodes']], x[record['contextual_donors']])
        self.assertEqual(len(set(record['structural_nodes']) & set(record['contextual_nodes'])), 0)
        self.assertGreaterEqual(np.diff(record['clique_offsets']).min(), 2)
        second = inject_anomalies(x, adj, ratio=0.22, clique_size=5, candidates=8)
        np.testing.assert_array_equal(second[0], modified)
        np.testing.assert_array_equal(second[2], y)
        stronger = inject_anomalies(x, adj, ratio=0.22, clique_size=12, candidates=8)
        np.testing.assert_array_equal(stronger[0], modified)
        np.testing.assert_array_equal(stronger[2], y)
        for key in splits:
            np.testing.assert_array_equal(stronger[3][key], splits[key])
        self.assertGreater(stronger[1].nnz, graph.nnz)

    def test_kg_multihop_and_center_not_an_attribute(self):
        kg = SequenceKG(np.array([[10, 1, 20], [20, 2, 30], [99, 3, -1]]), np.array([1, 2, 3]))
        path = kg.shortest_paths(np.array([50, 100, 10]), 100, [2, 3])
        self.assertEqual(path[2]['entities'], [100, 1, 2])
        self.assertEqual(path[2]['attributes'], [10, 20])
        self.assertFalse(path[3]['reachable'])
        self.assertFalse(kg.shortest_paths(np.array([1, 100, -1]), 100, [1])[1]['reachable'])

    def test_kg_duplicate_attributes_do_not_overflow(self):
        kg = SequenceKG(np.tile([10, 1, 20], (512, 1)), np.ones(512, np.int64))
        self.assertTrue(kg.shortest_paths(np.array([10, 5, -1]), 5, [1])[1]['reachable'])

    def test_memory_eviction_new_admission_and_feedback(self):
        memory = AdaptiveMemory(4, capacity=2)
        np.testing.assert_array_equal(memory.weights(np.arange(4)), np.ones(4))
        memory.retrieve([0, 1])
        memory.supervised_feedback([1], True)
        self.assertEqual(memory.utility[1], 1)
        before = memory.utility.copy()
        memory.retrieve([1])
        np.testing.assert_array_equal(memory.utility, before)
        memory.retrieve([2])
        self.assertEqual(memory.active.tolist(), [False, True, True, False])
        self.assertEqual(memory.weights(np.array([0]))[0], 1.0)
        memory.retrieve([0])
        self.assertEqual(memory.count, 2)

    def test_equation19_decays_the_entire_active_weight(self):
        memory = AdaptiveMemory(2)
        memory.retrieve([0])
        memory.frequency[0] = 10
        memory.utility[0] = 2
        memory.clock = int(memory.last[0]) + 100
        expected = (2 + 0.1 * np.log(11)) / np.e
        self.assertAlmostEqual(memory.weights([0])[0], expected, places=12)
        self.assertLess(memory.weights([0])[0], 1.0)
        self.assertEqual(memory.weights([1])[0], 1.0)
        memory.clock += 10000
        self.assertLess(memory.weights([0])[0], 1e-40)
        self.assertEqual(memory.weights([1])[0], 1.0)

    def test_stale_memory_does_not_outrank_an_identical_fresh_reference(self):
        z = np.array([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=np.float32)
        mask = np.ones((2, 1), bool)
        retriever = EntropyRetriever({'z': z, 'mask': mask, 'target': np.arange(2)}, topk=1)
        retriever.memory.retrieve([0])
        retriever.memory.frequency[0] = 10
        retriever.memory.utility[0] = 2
        retriever.memory.clock += 100
        refs, _ = retriever.query_batch(z[:1], mask[:1])
        np.testing.assert_array_equal(refs, [[1]])

    def test_exact_pruning_handles_memory_weights_below_one(self):
        rng = np.random.default_rng(91)
        z = rng.normal(size=(320, 7, 8)).astype(np.float32)
        mask = np.ones((320, 7), bool)
        bank = {'z': z, 'mask': mask, 'target': np.arange(320)}
        pruned = EntropyRetriever(bank, candidates=0)
        exhaustive = EntropyRetriever(bank, candidates=320)
        for retriever in (pruned, exhaustive):
            retriever.memory.retrieve(np.arange(80))
            retriever.memory.frequency[:80] = 10
            retriever.memory.utility[:80] = 2
            retriever.memory.clock += 100
        actual = pruned.query_batch(z[:5], mask[:5])
        expected = exhaustive.query_batch(z[:5], mask[:5])
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1], atol=0, rtol=0)

    def test_exhaustive_retrieval_searches_outside_memory(self):
        z = np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 0.0]]], dtype=np.float32)
        bank = {'z': z, 'mask': np.ones((3, 1), bool), 'target': np.arange(3)}
        retriever = EntropyRetriever(bank, candidates=0, topk=1, capacity=1)
        refs, _ = retriever.query_batch(z, bank['mask'])
        np.testing.assert_array_equal(refs.ravel(), [0, 1, 2])
        self.assertEqual(retriever.memory.count, 1)

    def test_exact_bound_pruning_matches_exhaustive_ranking(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=(240, 13, 8)).astype(np.float32)
        mask = np.ones((240, 13), bool)
        mask[::2, 9:] = False
        bank = {'z': z, 'mask': mask, 'target': np.arange(len(z))}
        a = EntropyRetriever(bank, candidates=0, topk=5, capacity=13)
        b = EntropyRetriever(bank, candidates=len(z), topk=5, capacity=13)
        for i in range(8):
            ia, sa = a.query_batch(z[i:i + 1], mask[i:i + 1])
            ib, sb = b.query_batch(z[i:i + 1], mask[i:i + 1])
            np.testing.assert_array_equal(ia, ib)
            np.testing.assert_allclose(sa, sb)
            a.memory.supervised_feedback(ia.ravel(), True)
            b.memory.supervised_feedback(ib.ravel(), True)
        restored = AdaptiveMemory(len(z), capacity=13)
        restored.load_state_dict(a.memory.state_dict())
        np.testing.assert_array_equal(restored.weights(np.arange(len(z))), a.memory.weights(np.arange(len(z))))
        bank['target'] = np.repeat(np.arange(120), 2)
        a = EntropyRetriever(bank, candidates=0, topk=5, capacity=180)
        b = EntropyRetriever(bank, candidates=len(z), topk=5, capacity=180)
        for retriever in [a, b]:
            retriever.memory.retrieve(np.arange(180))
            retriever.memory.utility[:16] = 100
            retriever.memory.utility[16:150] = 2
        ia, sa = a.query_batch(z[:12], mask[:12], excluded_centers=bank['target'][:12])
        ib, sb = b.query_batch(z[:12], mask[:12], excluded_centers=bank['target'][:12])
        np.testing.assert_array_equal(ia, ib)
        np.testing.assert_allclose(sa, sb, atol=0, rtol=0)
        for key in ['active', 'frequency', 'last', 'utility']:
            np.testing.assert_array_equal(getattr(a.memory, key), getattr(b.memory, key))

    def test_frozen_quantized_linear_input_gradient(self):
        torch.manual_seed(5)
        qweight = torch.randint(-2 ** 30, 2 ** 30, (16, 2), dtype=torch.int32)
        qzeros = torch.randint(-2 ** 30, 2 ** 30, (2, 2), dtype=torch.int32)
        scales = torch.rand(2, 16)
        layer = FrozenAWQLinear(qweight, qzeros, scales)
        x = torch.randn(3, 16, requires_grad=True)
        expected = dequantize_reference(qweight, qzeros, scales)
        out = layer(x)
        torch.testing.assert_close(out, x @ expected)
        out.sum().backward()
        torch.testing.assert_close(x.grad, expected.sum(-1).expand_as(x))
        self.assertEqual(len(list(layer.parameters())), 0)

    def test_vq_straight_through_updates_encoder_but_is_discrete(self):
        tokenizer = GraphTokenizer(np.eye(4, dtype=np.float32), 8)
        z = torch.randn(2, 13, 4, requires_grad=True)
        mask = torch.ones(2, 13, dtype=torch.bool)
        tokens, codes, loss = tokenizer(z, mask)
        self.assertEqual(codes.dtype, torch.int64)
        (tokens.square().sum() + loss).backward()
        self.assertGreater(float(z.grad.norm()), 0)
        self.assertIsNone(tokenizer.codebook.grad)
        self.assertFalse(tokenizer.codebook.requires_grad)
        self.assertGreater(float(tokenizer.project.weight.grad.norm()), 0)
if __name__ == '__main__':
    unittest.main()
