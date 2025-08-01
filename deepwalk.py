import numpy as np
import networkx as nx
import random
from tqdm import tqdm


def read_edge(edge_txt):
    edges = np.loadtxt(edge_txt, dtype=np.int16)
    edges = [(edges[i, 0], edges[i, 1]) for i in range(edges.shape[0])]
    return edges


class SCRADSequenceConstructor:
    def __init__(self, node_num, edge_index, undirected=True) -> None:
        if undirected:
            self.G = nx.Graph()
        else:
            self.G = nx.DiGraph()
        self.G.add_nodes_from(list(range(node_num)))
        
    
        edge_list = [tuple(edge_index[:, i].tolist()) for i in range(edge_index.shape[1])]
        self.G.add_edges_from(edge_list)
        
    
        self.neighbors = {node: list(self.G.neighbors(node)) for node in self.G.nodes}
        self.node_num = node_num

    def build_bidirectional_sequence(self, target_node, seq_length):
        if seq_length % 2 == 0:
            raise ValueError("Sequence length must be odd to ensure target node is in the center")
        half_length = (seq_length - 1) // 2
        
     
        forward = [target_node]
        current = target_node
        for _ in range(half_length):
            if not self.neighbors[current]:  
                break
            # Avoid cycles
            candidates = [n for n in self.neighbors[current] if n not in forward]
            if not candidates:
                break
            next_node = random.choice(candidates)
            forward.append(next_node)
            current = next_node
        
       
        backward = []
        current = target_node
        for _ in range(half_length):
            if not self.neighbors[current]: 
                break
           
            candidates = [n for n in self.neighbors[current] if n not in forward and n not in backward]
            if not candidates:
                break
            next_node = random.choice(candidates)
            backward.append(next_node)
            current = next_node
        
      
        backward.reverse()
        full_sequence = backward + forward
       
        if len(full_sequence) > seq_length:
            full_sequence = full_sequence[:seq_length]
        while len(full_sequence) < seq_length:
            full_sequence.append(random.choice(list(self.G.nodes)))  
        return [str(node) for node in full_sequence]

    def build_total_sequences(self, num_sequences_per_node, seq_length):
        print("Starting bidirectional sequence construction...")
        total_sequences = []
        self.node_sequences = {node: [] for node in self.G.nodes}  
        
        for node in tqdm(self.G.nodes, desc="Processing nodes"):
            for _ in range(num_sequences_per_node):
                seq = self.build_bidirectional_sequence(node, seq_length)
                total_sequences.append(seq)
                self.node_sequences[node].append(seq)
        
        return total_sequences

    def concate(self, sequences, node_num, seq_num, length):
        graphseq = []
        node_sequences_dict = {}
        node_sequences_index_dict = {}
        
        for i in range(node_num):
            if i not in self.node_sequences:
                continue
            valid_sequences = []
            temp_indices = []
           
            for seq in self.node_sequences[i][:seq_num]:
                if len(seq) >= length:
                    valid_sequences.append(seq[:length])
                    temp_indices.append(len(graphseq) + len(valid_sequences) - 1)
            
            if valid_sequences:
                node_sequences_dict[i] = valid_sequences
                node_sequences_index_dict[i] = temp_indices
                graphseq.extend(valid_sequences)
        
        return graphseq, node_sequences_index_dict