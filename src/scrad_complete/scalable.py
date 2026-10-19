import numpy as np
from scipy import sparse
import torch

def original_partition_adjacency(adj, splits):
    membership = np.full(adj.shape[0], -1, np.int8)
    for index, ids in enumerate(splits.values()):
        if np.any(membership[ids] >= 0):
            raise ValueError('Original partitions overlap')
        membership[ids] = index
    if np.any(membership < 0):
        raise ValueError('Original partitions do not cover every node')
    result = adj.tocsr(copy=True)
    rows = np.repeat(membership, np.diff(result.indptr))
    result.data[rows != membership[result.indices]] = 0
    result.eliminate_zeros()
    return result

class _SymmetricSparseMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, adj, h):
        ctx.adj = adj
        return torch.sparse.mm(adj, h)

    @staticmethod
    def backward(ctx, grad):
        return (None, torch.sparse.mm(ctx.adj, grad))

def csr_tensor(adj, device):
    adj = adj.tocsr()
    return torch.sparse_csr_tensor(torch.as_tensor(adj.indptr, device=device), torch.as_tensor(adj.indices, device=device), torch.as_tensor(adj.data, device=device), size=adj.shape, device=device)

def pmp_unknown(model, x, adj):
    h = model.project(x)
    for layer in model.layers:
        unknown = _SymmetricSparseMM.apply(adj, h)
        zeros = torch.zeros_like(h)
        balance = layer.balance(h)
        neighbors = layer.benign(zeros, h) + layer.fraud(zeros, h)
        neighbors = neighbors + balance * layer.fraud(unknown, h) + (1 - balance) * layer.benign(unknown, h)
        h = torch.relu(layer.norm(layer.self_linear(h) + neighbors))
    return (h, model.classifier(h).squeeze(-1))

class MappedEdgeScales:

    def __init__(self, array, device):
        self.array, self.device = (array, device)
        self.cached_ids = None
        self.cached_values = None

    def prefetch(self, rows, max_bytes=2 * 1024 ** 3, chunk_size=32768):
        ids = np.unique(np.asarray(rows).clip(0))
        if len(ids) and ids[-1] >= len(self.array):
            raise ValueError('Cached edge ID is outside the source bank')
        required = len(ids) * int(np.prod(self.array.shape[1:])) * self.array.dtype.itemsize
        if not len(ids) or required > max_bytes:
            return 0
        values = np.empty((len(ids), *self.array.shape[1:]), self.array.dtype)
        for start in range(0, len(ids), chunk_size):
            selected = ids[start:start + chunk_size]
            values[start:start + len(selected)] = self.array[selected]
        self.cached_ids, self.cached_values = (ids, values)
        return values.nbytes

    def __getitem__(self, index):
        rows = index.detach().cpu().numpy()
        if self.cached_ids is not None:
            flat = rows.ravel()
            positions = np.searchsorted(self.cached_ids, flat).clip(0, len(self.cached_ids) - 1)
            hits = self.cached_ids[positions] == flat
            result = np.empty((len(flat), *self.array.shape[1:]), self.array.dtype)
            result[hits] = self.cached_values[positions[hits]]
            if not hits.all():
                result[~hits] = self.array[flat[~hits]]
            return torch.as_tensor(result.reshape((*rows.shape, *self.array.shape[1:])), device=self.device)
        return torch.as_tensor(np.asarray(self.array[rows]), device=self.device)

def fast_sample_sequences(part, centers, rng, length=7):
    if length % 2 != 1:
        raise ValueError('Sequence length must be odd')
    centers = np.asarray(centers)
    count, mid = (len(centers), length // 2)
    seq = np.full((count, length), -1, np.int32)
    seq[:, mid] = centers
    degree = np.diff(part.adj.indptr)
    for direction in [-1, 1]:
        current = centers.copy()
        alive = np.ones(count, bool)
        for distance in range(1, mid + 1):
            alive &= degree[current] > 0
            pending = np.flatnonzero(alive)
            chosen = np.full(count, -1, np.int32)
            for _ in range(8):
                if not len(pending):
                    break
                node = current[pending]
                offsets = (rng.random(len(pending)) * degree[node]).astype(np.int64)
                neighbor = part.adj.indices[part.adj.indptr[node] + offsets]
                valid = ~(seq[pending] == neighbor[:, None]).any(1)
                chosen[pending[valid]] = neighbor[valid]
                pending = pending[~valid]
            for row in pending:
                node = current[row]
                candidates = part.adj.indices[part.adj.indptr[node]:part.adj.indptr[node + 1]]
                candidates = candidates[~np.isin(candidates, seq[row])]
                if len(candidates):
                    chosen[row] = rng.choice(candidates)
            alive &= chosen >= 0
            seq[:, mid + direction * distance] = chosen
            current[alive] = chosen[alive]
    valid = (seq[:, :-1] >= 0) & (seq[:, 1:] >= 0)
    edges = np.full((count, length - 1), -1, np.int32)
    edges[valid] = np.asarray(part.edge_lookup[seq[:, :-1][valid], seq[:, 1:][valid]]).ravel() - 1
    if (edges[valid] < 0).any():
        raise ValueError('Sampled transition is not a graph edge')
    return (seq, edges)

def selected_line_graph_scales(part, node_h, selected_edges, depth=3, restart=0.1, device='cpu', chunk_size=65536, output=None):
    if depth < 1 or not 0 <= restart <= 1:
        raise ValueError('Positive depth and restart in [0,1] are required')
    selected_edges = np.asarray(selected_edges, np.int64)
    if (selected_edges < 0).any() or (selected_edges >= len(part.edges)).any():
        raise ValueError('Invalid selected edge ID')
    h = torch.as_tensor(node_h, dtype=torch.float32, device=device)
    degree = np.diff(part.adj.indptr).astype(np.float32)
    rows = np.repeat(np.arange(len(degree), dtype=np.int32), np.diff(part.adj.indptr))
    d = 1 / np.maximum(degree[rows] + degree[part.adj.indices] - 1, 1)
    row_device = torch.as_tensor(rows, device=device, dtype=torch.long)
    d_device = torch.as_tensor(d, device=device)
    ptr = torch.as_tensor(part.adj.indptr, device=device)
    col = torch.as_tensor(part.adj.indices, device=device)

    def multiply(power, vectors):
        weights = d_device.pow(power)
        matrix = torch.sparse_csr_tensor(ptr, col, weights, size=part.adj.shape, device=device)
        diagonal = h.new_zeros(len(h)).scatter_add_(0, row_device, weights)
        return torch.sparse.mm(matrix, vectors) + diagonal[:, None] * vectors
    a, b = (1 - restart, restart)
    with torch.no_grad():
        terms = [0.5 * multiply(j + 0.5, h) for j in range(depth)]
        aggregates = [terms[0]]
        for level in range(1, depth):
            value = (-a) ** level * terms[level]
            for j in range(level):
                value = value + a * (-a) ** j * multiply(j + 1, aggregates[level - 1 - j])
                value = value + b * (-a) ** j * terms[j]
            aggregates.append(value)
        shape = (len(selected_edges), depth, h.shape[1])
        result = np.empty(shape, np.float32) if output is None else output
        if result.shape != shape:
            raise ValueError('Output shape mismatch')
        for start in range(0, len(selected_edges), chunk_size):
            endpoints = part.edges[selected_edges[start:start + chunk_size]]
            u, v = (torch.as_tensor(endpoints[:, i], device=device) for i in [0, 1])
            inv = torch.as_tensor(1 / np.sqrt(np.maximum(degree[endpoints[:, 0]] + degree[endpoints[:, 1]] - 1, 1)), device=device)
            initial = 0.5 * (h[u] + h[v])
            w0 = inv[:, None] * initial
            current = w0
            for level in range(depth):
                current = a * inv[:, None].square() * (aggregates[level][u] + aggregates[level][v] - current) + b * w0
                result[start:start + len(u), level] = (current / inv[:, None]).cpu().numpy()
    return result
