import torch
from torch import nn
from torch.nn import functional as F

def sparse_tensor(a, device):
    a = a.tocoo()
    import numpy as np
    return torch.sparse_coo_tensor(torch.tensor(np.stack([a.row, a.col]), device=device), torch.tensor(a.data, device=device), a.shape).coalesce()

class ConditionalLinear(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.condition = nn.Linear(d, d)
        self.linear = nn.Linear(d, d)

    def forward(self, neighbors, center):
        return self.linear(neighbors * self.condition(center))

class PMPLayer(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.benign = ConditionalLinear(d)
        self.fraud = ConditionalLinear(d)
        self.self_linear = nn.Linear(d, d)
        self.balance = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1), nn.Sigmoid())
        self.norm = nn.LayerNorm(d)

    def forward(self, h, adj, known):
        sums = [torch.sparse.mm(adj, h * (known == label).unsqueeze(-1)) for label in (0, 1, 2)]
        b = self.balance(h)
        neighbors = self.benign(sums[0], h) + self.fraud(sums[1], h)
        neighbors = neighbors + b * self.fraud(sums[2], h) + (1 - b) * self.benign(sums[2], h)
        return F.relu(self.norm(self.self_linear(h) + neighbors))

class PMPEncoder(nn.Module):

    def __init__(self, input_dim, d=64):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, d), nn.ReLU(), nn.LayerNorm(d))
        self.layers = nn.ModuleList([PMPLayer(d), PMPLayer(d)])
        self.classifier = nn.Linear(d, 1)

    def forward(self, x, adj, known):
        h = self.project(x)
        for layer in self.layers:
            h = layer(h, adj, known)
        return (h, self.classifier(h).squeeze(-1))

class DualEdgeEncoder(nn.Module):

    def __init__(self, d=64, depth=3, local_hops=2, alpha=0.5, paper=False):
        super().__init__()
        self.alpha, self.local_hops = (alpha, local_hops)
        self.paper = paper
        if paper:
            self.left_hops = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(local_hops)])
            self.right_hops = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(local_hops)])
        self.left = nn.Linear(d, d)
        self.right = nn.Linear(d, d)
        self.direction_score = nn.Linear(d, 1, bias=not paper)
        self.global_encoder = nn.Linear(d, d, bias=False)
        self.scale_maps = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(depth)])
        self.initial_map = nn.Linear(d, d, bias=False)
        self.scale_score = nn.Linear(d, 1, bias=False)
        self.gate = nn.Parameter(torch.zeros(d))
        self.norm = nn.LayerNorm(d)
        self.coherence_head = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))
        if paper:
            for module in (self.left, self.right, self.norm, self.coherence_head):
                module.requires_grad_(False)

    def forward(self, node_h, global_scales, seq, edge_ids):
        nm = seq >= 0
        em = edge_ids >= 0
        nodes = node_h[seq.clamp_min(0)] * nm.unsqueeze(-1)
        base = (self.alpha * nodes[:, :-1] + (1 - self.alpha) * nodes[:, 1:]) * em.unsqueeze(-1)
        left, right = (base.clone(), base.clone())
        if self.paper:
            for wl, wr in zip(self.left_hops, self.right_hops):
                left = wl(left + F.pad(left[:, :-1], (0, 0, 1, 0))) * em.unsqueeze(-1)
                right = wr(right + F.pad(right[:, 1:], (0, 0, 0, 1))) * em.unsqueeze(-1)
            sides = torch.stack([left, right], 2)
            attn = F.softmax(torch.tanh(self.direction_score(sides)), dim=2)
        else:
            for offset in range(1, self.local_hops + 1):
                l = F.pad(base[:, :-offset], (0, 0, offset, 0))
                r = F.pad(base[:, offset:], (0, 0, 0, offset))
                left = left + l / (offset + 1)
                right = right + r / (offset + 1)
            sides = torch.stack([self.left(left), self.right(right)], 2)
            attn = F.softmax(self.direction_score(torch.tanh(sides)), dim=2)
        local = (sides * attn).sum(2)
        scales = global_scales[edge_ids.clamp_min(0)]
        encoded = F.relu(self.global_encoder(scales))
        init = self.initial_map(base)
        logits = torch.stack([self.scale_score(torch.tanh(layer(encoded[:, :, i]) + init)).squeeze(-1) for i, layer in enumerate(self.scale_maps)], -1)
        global_h = (encoded * logits.softmax(-1).unsqueeze(-1)).sum(2)
        g = self.gate.sigmoid()
        fused = g * local + (1 - g) * global_h
        edges = (fused if self.paper else self.norm(fused)) * em.unsqueeze(-1)
        elements = nodes.new_zeros(len(nodes), 2 * nodes.shape[1] - 1, nodes.shape[2])
        mask = torch.zeros(elements.shape[:2], dtype=torch.bool, device=elements.device)
        elements[:, ::2], elements[:, 1::2] = (nodes, edges)
        mask[:, ::2], mask[:, 1::2] = (nm, em)
        pooled = (elements * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        mismatch = (nodes[:, :-1] + nodes[:, 1:]) / 2 - edges
        mismatch = (mismatch.abs() * em.unsqueeze(-1)).sum(1) / em.sum(1, keepdim=True).clamp_min(1)
        center = nodes[:, nodes.shape[1] // 2]
        logit = self.coherence_head(torch.cat([center, pooled, mismatch], -1)).squeeze(-1)
        return (elements, mask, logit)
