import numpy as np
from scipy.optimize import linear_sum_assignment

def entropy_similarity(a, b):
    if not len(a) or not len(b):
        return 0.0
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-08)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-08)
    if len(a) < len(b):
        a = np.tile(a, (int(np.ceil(len(b) / len(a))), 1))
    elif len(b) < len(a):
        b = np.tile(b, (int(np.ceil(len(a) / len(b))), 1))
    sim = np.clip((a @ b.T + 1) * 0.5, 0, 1)
    row, col = linear_sum_assignment(sim, maximize=True)
    s = sim[row, col]
    if s.sum() <= 1e-12:
        return 0.0
    if len(s) == 1:
        entropy = 0.0
    else:
        p = s / s.sum()
        p = p[p > 0]
        entropy = -float(np.sum(p * np.log(p))) / np.log(len(s))
    return float(np.mean(s) * (1 + entropy))

def pooled(z, mask):
    h = (z * mask[..., None]).sum(1) / mask.sum(1, keepdims=True).clip(1)
    return h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-08)

def normalized_matching_scores(query, candidates):
    n, m = (query.shape[0], candidates.shape[1])
    if n < m:
        query = np.tile(query, (int(np.ceil(m / n)), 1))
    elif m < n:
        candidates = np.tile(candidates, (1, int(np.ceil(n / m)), 1))
    similarity = np.clip((np.matmul(query, candidates.transpose(0, 2, 1)) + 1) * 0.5, 0, 1)
    matches = []
    for matrix in similarity:
        row, col = linear_sum_assignment(matrix, maximize=True)
        matches.append(matrix[row, col])
    scores = np.stack(matches)
    if scores.shape[1] == 1:
        return scores.mean(1)
    probs = scores / np.maximum(scores.sum(1, keepdims=True), 1e-12)
    entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(1) / np.log(scores.shape[1])
    return scores.mean(1) * (1 + entropy)

class Retriever:

    def __init__(self, bank, candidates=128, topk=5, lambdas=(0.1, 0.5, 0.01)):
        self.z = bank['z']
        self.mask = bank['mask']
        self.pool = pooled(self.z, self.mask)
        self.lengths = self.mask.sum(1)
        norm = self.z / np.maximum(np.linalg.norm(self.z, axis=-1, keepdims=True), 1e-08)
        self.valid = [z[m] for z, m in zip(norm, self.mask)]
        self.utility = bank['utility'].copy()
        self.frequency = np.zeros(len(self.z), np.int64)
        self.last = np.zeros(len(self.z), np.int64)
        self.topk = topk
        self.candidates = min(candidates, len(self.z)) if candidates else len(self.z)
        self.lambdas = lambdas
        self.clock = 0

    def query_batch(self, z, mask):
        similarities = pooled(z, mask) @ self.pool.T
        results = []
        scores = []
        for item, (zz, mm, sim) in enumerate(zip(z, mask, similarities)):
            l1, l2, l3 = self.lambdas
            weights = 1 + l1 * np.log1p(self.frequency) + l2 * self.utility * np.exp(-l3 * (self.clock - self.last))
            if self.candidates < len(sim):
                cand = np.argpartition(sim, -self.candidates)[-self.candidates:]
            else:
                cand = np.arange(len(sim))
            query = zz[mm]
            query = query / np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), 1e-08)
            raw = np.empty(len(cand), np.float32)
            for length in np.unique(self.lengths[cand]):
                positions = np.flatnonzero(self.lengths[cand] == length)
                batch = np.stack([self.valid[i] for i in cand[positions]])
                raw[positions] = normalized_matching_scores(query, batch)
            ranks = raw * weights[cand]
            order = np.argsort(-ranks, kind='stable')[:self.topk]
            idx = cand[order]
            self.frequency[idx] += 1
            self.last[idx] = self.clock
            self.clock += 1
            results.append(idx)
            scores.append(ranks[order])
        return (np.array(results, np.int32), np.array(scores, np.float32))

def shared_path(query_nodes, reference_nodes):
    shared = sorted(set(query_nodes[query_nodes >= 0].tolist()) & set(reference_nodes[reference_nodes >= 0].tolist()))
    return {'reachable': bool(shared), 'shared_nodes': shared}
