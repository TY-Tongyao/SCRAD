# deepwalk
import numpy as np
import networkx as nx
import random
from tqdm import tqdm

def read_edge(edge_txt):
    edges = np.loadtxt(edge_txt, dtype=np.int64)
    return [(int(u), int(v)) for u, v in edges]

class deepwalk_hy:
    def __init__(self, node_num, edge_index, undirected=False):

        self.G = nx.Graph() if undirected else nx.DiGraph()
        self.G.add_nodes_from(range(node_num))
        edges = [(int(edge_index[0,i]), int(edge_index[1,i])) 
                 for i in range(edge_index.shape[1])]
        self.G.add_edges_from(edges)
        
        self.neighbors = {n: list(self.G.neighbors(n)) for n in range(node_num)}

    def random_walk(self, length, alpha=0.0, seed=42, start=None):
        
        rand = random.Random(seed)
        cur = start if start is not None else rand.choice(list(self.neighbors.keys()))
        path = [cur]
        while len(path) < length:
            nbrs = self.neighbors.get(cur, [])
            if nbrs and rand.random() >= alpha:
                cur = rand.choice(nbrs)
            else:
                cur = path[0]  # 重启
            path.append(cur)
        return [str(x) for x in path]

    def build_total_corpus(self, num_rounds, walk_length, alpha=0.0):
        self.node_sequences = {n: [] for n in self.neighbors}
        all_walks = []
        for r in tqdm(range(num_rounds), desc="DeepWalk rounds"):
            for n in self.neighbors:
                w = self.random_walk(walk_length, alpha, seed=r, start=n)
                all_walks.append(w)
                self.node_sequences[n].append(w)
        return all_walks

    def concate(self, walks, node_num, seq_num, min_length):
        graphseq = []
        idx_map = {}
        for n in range(node_num):
            seqs = []
            idxs = []
            if len(self.node_sequences[n]) < seq_num*2:
                continue
            for j in range(seq_num):
                fwd = self.node_sequences[n][j][:-1]
                bwd = self.node_sequences[n][j+seq_num]
                bwd_rev = list(reversed(bwd))
                seq = bwd_rev + fwd
                if len(seq) < min_length:
                    seqs = []
                    break
                idxs.append(len(graphseq) + len(seqs))
                seqs.append(seq)
            if seqs:
                idx_map[n] = idxs
                graphseq.extend(seqs)
        return graphseq, idx_map
