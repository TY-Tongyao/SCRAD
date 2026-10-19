import heapq
import numpy as np
from scrad.retrieval import normalized_matching_scores, pooled

class AdaptiveMemory:

    def __init__(self, size, capacity=10000, lambdas=(0.1, 0.5, 0.01)):
        self.capacity = min(int(capacity), int(size))
        if self.capacity < 1:
            raise ValueError('Memory capacity must be positive')
        self.active = np.zeros(size, bool)
        self.frequency = np.zeros(size, np.int64)
        self.last = np.zeros(size, np.int64)
        self.utility = np.zeros(size, np.float64)
        self.lambdas = lambdas
        self.clock = 0
        self.heap = []
        self.count = 0

    def weights(self, ids):
        ids = np.asarray(ids)
        a, b, c = self.lambdas
        values = (1 + a * np.log1p(self.frequency[ids]) + b * self.utility[ids]) * np.exp(-c * (self.clock - self.last[ids]))
        return np.where(self.active[ids], values, 1.0)

    def retrieve(self, ids):
        self.clock += 1
        for index in dict.fromkeys((int(x) for x in ids)):
            if not self.active[index]:
                if self.count == self.capacity:
                    while self.heap:
                        timestamp, victim = heapq.heappop(self.heap)
                        if self.active[victim] and self.last[victim] == timestamp:
                            self.active[victim] = False
                            self.count -= 1
                            break
                self.active[index] = True
                self.frequency[index] = 0
                self.utility[index] = 0
                self.count += 1
            self.frequency[index] += 1
            self.last[index] = self.clock
            heapq.heappush(self.heap, (self.clock, index))
        if len(self.heap) > 4 * self.capacity:
            self.heap = [(int(self.last[i]), int(i)) for i in np.flatnonzero(self.active)]
            heapq.heapify(self.heap)

    def supervised_feedback(self, ids, correct):
        if correct:
            for index in set((int(x) for x in ids)):
                if self.active[index]:
                    self.utility[index] += 1.0

    def state_dict(self):
        import torch
        return {key: torch.as_tensor(getattr(self, key).copy()) for key in ['active', 'frequency', 'last', 'utility']} | {'clock': self.clock}

    def load_state_dict(self, state):
        for key in ['active', 'frequency', 'last', 'utility']:
            value = state[key].cpu().numpy().copy()
            if value.shape != self.active.shape:
                raise ValueError('Memory state/corpus size mismatch')
            setattr(self, key, value)
        self.clock = int(state['clock'])
        self.count = int(self.active.sum())
        self.heap = [(int(self.last[i]), int(i)) for i in np.flatnonzero(self.active)]
        heapq.heapify(self.heap)

class EntropyRetriever:

    def __init__(self, bank, candidates=0, topk=5, capacity=10000, device='cpu'):
        self.bank = bank
        self.z = np.asarray(bank['z'], dtype=np.float32)
        self.mask = bank['mask']
        self.lengths = self.mask.sum(1)
        if not np.all(self.lengths > 0):
            raise ValueError('Empty reference sequence')
        self.norm = np.empty_like(self.z)
        self.pool = None
        self.candidates = int(candidates)
        self.topk = int(topk)
        self.memory = AdaptiveMemory(len(self.z), capacity)
        self.device = device
        self.pool_device = None
        self.update_vectors(self.z)

    def update_vectors(self, vectors):
        values = np.asarray(vectors, dtype=np.float32)
        if values.shape != self.z.shape:
            raise ValueError('Reference vector shape changed')
        self.z = values
        for start in range(0, len(values), 2048):
            stop = start + 2048
            block = self.z[start:stop]
            np.divide(block, np.maximum(np.linalg.norm(block, axis=-1, keepdims=True), 1e-08), out=self.norm[start:stop])
            rows = pooled(block, self.mask[start:stop])
            if self.pool is None:
                self.pool = np.empty((len(values), values.shape[-1]), dtype=rows.dtype)
            self.pool[start:stop] = rows
        if self.device != 'cpu':
            import torch
            self.pool_device = torch.as_tensor(self.pool, device=self.device)

    def query_batch(self, z, mask, excluded_centers=None):

        def cosine_for(values, valid):
            if self.pool_device is not None:
                import torch
                return (torch.as_tensor(pooled(values, valid), device=self.device) @ self.pool_device.T).cpu().numpy()
            return pooled(values, valid) @ self.pool.T
        cosine = cosine_for(z, mask) if self.candidates else None
        references, scores = ([], [])
        for row, (query, valid) in enumerate(zip(z, mask)):
            query = query[valid]
            query = query / np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), 1e-08)

            def score(selected):
                raw = np.empty(len(selected), np.float32)
                for length in np.unique(self.lengths[selected]):
                    group = np.flatnonzero(self.lengths[selected] == length)
                    for offset in range(0, len(group), 2048):
                        positions = group[offset:offset + 2048]
                        batch = np.stack([self.norm[i, self.mask[i]] for i in selected[positions]])
                        raw[positions] = normalized_matching_scores(query, batch)
                return raw * self.memory.weights(selected)
            if not self.candidates:
                active = np.flatnonzero(self.memory.active)
                if excluded_centers is not None:
                    active = active[self.bank['target'][active] != excluded_centers[row]]
                upper = (2.0 + 1e-05) * self.memory.weights(active)
                order = np.lexsort((active, -upper))
                ranked, upper = (active[order], upper[order])
                chosen, chosen_scores = (np.empty(0, np.int64), np.empty(0, np.float64))
                for offset in range(0, len(ranked), 64):
                    threshold = chosen_scores[-1] if len(chosen) == self.topk else -np.inf
                    if upper[offset] < threshold:
                        break
                    current = ranked[offset:offset + 64]
                    current = current[upper[offset:offset + 64] >= threshold]
                    merged = np.r_[chosen, current]
                    values = np.r_[chosen_scores, score(current)]
                    order = np.lexsort((merged, -values))[:self.topk]
                    chosen, chosen_scores = (merged[order], values[order])
                if len(chosen) == self.topk and chosen_scores[-1] > 2.0 + 1e-05:
                    self.memory.retrieve(chosen)
                    references.append(chosen)
                    scores.append(chosen_scores)
                    continue
            sim = cosine[row] if cosine is not None else cosine_for(z[row:row + 1], mask[row:row + 1])[0]
            eligible = np.ones(len(self.z), bool)
            if excluded_centers is not None:
                eligible &= self.bank['target'] != excluded_centers[row]
            ids = np.flatnonzero(eligible)
            if len(ids) < self.topk:
                raise ValueError('Too few eligible reference sequences')
            if self.candidates and self.candidates < len(ids):
                ids = ids[np.argpartition(sim[ids], -self.candidates)[-self.candidates:]]
            if self.candidates:
                chosen, chosen_scores = (ids, score(ids))
            else:
                active = ids[self.memory.active[ids]]
                inactive = ids[~self.memory.active[ids]]
                first = inactive[np.argpartition(sim[inactive], -min(128, len(inactive)))[-min(128, len(inactive)):]] if len(inactive) else inactive
                warm = np.unique(np.r_[active, first])
                warm_scores = score(warm)
                order = np.lexsort((warm, -warm_scores))[:self.topk]
                chosen, chosen_scores = (warm[order], warm_scores[order])
                remaining = np.setdiff1d(ids, warm, assume_unique=True)
                bound = (2.0 + 1e-05) * self.memory.weights(remaining)
                remaining = remaining[bound >= chosen_scores[-1]]
                for offset in range(0, len(remaining), 2048):
                    current = remaining[offset:offset + 2048]
                    bound = (2.0 + 1e-05) * self.memory.weights(current)
                    current = current[bound >= chosen_scores[-1]]
                    if not len(current):
                        continue
                    merged = np.r_[chosen, current]
                    values = np.r_[chosen_scores, score(current)]
                    order = np.lexsort((merged, -values))[:self.topk]
                    chosen, chosen_scores = (merged[order], values[order])
            order = np.lexsort((chosen, -chosen_scores))[:self.topk]
            chosen, chosen_scores = (chosen[order], chosen_scores[order])
            self.memory.retrieve(chosen)
            references.append(chosen)
            scores.append(chosen_scores)
        return (np.asarray(references, np.int32), np.asarray(scores, np.float32))
