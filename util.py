from sklearn.metrics import roc_auc_score, f1_score, precision_score
import torch
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment

def evaluation(result, labels):
    result = result.cpu().detach().numpy() if isinstance(result, torch.Tensor) else result
    labels = labels.cpu().detach().numpy() if isinstance(labels, torch.Tensor) else labels
    
    auc = roc_auc_score(labels, result)
    
    num_anomalies = int(np.sum(labels))
    if num_anomalies == 0:
        return auc, 0.0, 0.0
    threshold_index = np.argsort(result)[-num_anomalies]
    threshold = result[threshold_index]
    binary_result = (result >= threshold).astype(int)
    
    f1 = f1_score(labels, binary_result)
    precision = precision_score(labels, binary_result) if num_anomalies > 0 else 0.0
    return auc, f1, precision

def get_sequence_indices(node_indices, graphseq_dict):
    seq_indices = []
    for node_index in node_indices:
        if node_index in graphseq_dict:
            seq_indices.extend(graphseq_dict[node_index])
    return seq_indices

def compute_ascore_majority(seq_result, graph_dict):
    seq_result = torch.tensor(seq_result, dtype=torch.float32)
    node_anomaly_scores = {}
    
    for node, seq_indices in graph_dict.items():
        seq_scores = seq_result[seq_indices]
        coherent_count = torch.sum(seq_scores < 0.5).item()
        incoherent_count = len(seq_scores) - coherent_count
        node_anomaly_scores[node] = incoherent_count / len(seq_scores)
    
    sorted_nodes = sorted(node_anomaly_scores.keys())
    return torch.tensor([node_anomaly_scores[node] for node in sorted_nodes], dtype=torch.float32)

def compute_dual_layer_edge_embeddings(node_embeddings, edge_index, adj, num_scales=3, gamma=0.1):
    src, dst = edge_index
    num_edges = edge_index.shape[1]
    
    # -------------------------- Local (Micro) Encoding --------------------------
    alpha = 0.5  
    local_emb = alpha * node_embeddings[src] + (1 - alpha) * node_embeddings[dst]
    
    edge_to_nodes = {i: (src[i].item(), dst[i].item()) for i in range(num_edges)}
    node_to_edges = {}
    for i, (u, v) in edge_to_nodes.items():
        if u not in node_to_edges:
            node_to_edges[u] = []
        if v not in node_to_edges:
            node_to_edges[v] = []
        node_to_edges[u].append(i)
        node_to_edges[v].append(i)
    
    W_forward = torch.nn.Parameter(torch.randn(local_emb.size(1), local_emb.size(1)))
    W_backward = torch.nn.Parameter(torch.randn(local_emb.size(1), local_emb.size(1)))
    
    agg_forward = []
    agg_backward = []
    for i in range(num_edges):
        u, v = edge_to_nodes[i]
        u_edges = [e for e in node_to_edges.get(u, []) if e != i]
        agg_u = torch.mean(local_emb[u_edges], dim=0) if u_edges else torch.zeros_like(local_emb[i])
        agg_forward.append(agg_u @ W_forward)
        
        v_edges = [e for e in node_to_edges.get(v, []) if e != i]
        agg_v = torch.mean(local_emb[v_edges], dim=0) if v_edges else torch.zeros_like(local_emb[i])
        agg_backward.append(agg_v @ W_backward)
    
    agg_forward = torch.stack(agg_forward)
    agg_backward = torch.stack(agg_backward)
    
    gate_weights = torch.sigmoid(torch.sum(agg_forward * agg_backward, dim=1, keepdim=True))
    local_final = gate_weights * agg_forward + (1 - gate_weights) * agg_backward
    
    # -------------------------- Global (Macro) Encoding --------------------------
    X = local_emb  
    A = adj
    I = torch.eye(A.size(0), device=A.device)
    A_tilde = A + I 
    D_tilde = torch.diag(torch.sum(A_tilde, dim=1))
    D_inv_sqrt = torch.inverse(torch.sqrt(D_tilde))
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt  
    
    scales = []
    current_X = X.T 
    for _ in range(num_scales):
        current_X = (1 - gamma) * (A_hat @ current_X.T).T + gamma * X.T
        scale_emb = torch.relu(current_X.T @ torch.nn.Parameter(torch.randn(X.size(1), X.size(1))))
        scales.append(scale_emb)
    
    scale_weights = torch.nn.functional.softmax(torch.stack([
        torch.sum(scale * local_final, dim=1) for scale in scales
    ]), dim=0)
    global_final = torch.sum(torch.stack([scale_weights[i] * scales[i].T for i in range(num_scales)]).T, dim=2)
    
    # -------------------------- Fusion --------------------------
    fusion_gate = torch.sigmoid(torch.nn.Linear(local_final.size(1), 1)(local_final))
    edge_embeddings = fusion_gate * local_final + (1 - fusion_gate) * global_final
    return edge_embeddings

def get_seq_with_edges(node_embeddings, edge_embeddings, edge_index, seq_lists):
    edge_dict = {}
    for j in range(edge_index.shape[1]):
        u, v = edge_index[0, j].item(), edge_index[1, j].item()
        edge_dict[(u, v)] = j
        edge_dict[(v, u)] = j  
    
    seq_data_list = []
    edge_data_list = []
    
    for seq in seq_lists:
        seq_nodes = [int(node) for node in seq]
        node_emb_list = [node_embeddings[node] for node in seq_nodes]
        
        edge_emb_list = []
        for i in range(len(seq_nodes) - 1):
            u, v = seq_nodes[i], seq_nodes[i+1]
            if (u, v) in edge_dict:
                edge_emb = edge_embeddings[edge_dict[(u, v)]]
                edge_emb_list.append(edge_emb)
            else:
                edge_emb_list.append(torch.zeros_like(edge_embeddings[0]))
        
        seq_data = torch.stack(node_emb_list)
        edge_data = torch.stack(edge_emb_list)
        
        seq_data_list.append(seq_data)
        edge_data_list.append(edge_data)
    
    return torch.stack(seq_data_list), torch.stack(edge_data_list)

def construct_coherence_label(node_labels, graph_seq):
    """Label sequences as coherent (0) or incoherent (1)"""
    seq_labels = []
    for seq in graph_seq:
        seq_nodes = [int(node) for node in seq]
        is_incoherent = any(node_labels[node] == 1 for node in seq_nodes)
        seq_labels.append(1.0 if is_incoherent else 0.0)
    return torch.tensor(seq_labels, dtype=torch.float32)

def entropy_constrained_similarity(seq_a, seq_b):
    sim_matrix = torch.nn.functional.cosine_similarity(
        seq_a.unsqueeze(1), seq_b.unsqueeze(0), dim=2
    ).numpy()

    row_ind, col_ind = linear_sum_assignment(-sim_matrix)  
    matched_sim = sim_matrix[row_ind, col_ind]
    
    if len(matched_sim) == 0:
        return 0.0
    norm_sim = matched_sim / np.sum(matched_sim)
    entropy = -np.sum(norm_sim * np.log(norm_sim + 1e-10))  
    normalized_entropy = entropy / np.log(len(matched_sim))  
    
    avg_sim = np.mean(matched_sim)
    return avg_sim * (1 + normalized_entropy)

def retrieve_top_k_sequences(target_seq, sequence_kg, all_sequences, k=5):
    similarities = []
    for idx, seq in enumerate(all_sequences):
        if seq.shape != target_seq.shape:
            continue
        sim = entropy_constrained_similarity(target_seq, seq)
        similarities.append((idx, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [idx for idx, _ in similarities[:k]]
    top_k_seqs = [all_sequences[idx] for idx in top_k_indices]

    top_k_paths = []
    target_seq_id = f"seq_{len(all_sequences)}"  
    for idx in top_k_indices:
        seq_id = f"seq_{idx}"
        try:
            path = nx.shortest_path(sequence_kg, source=target_seq_id, target=seq_id)
            top_k_paths.append(path)
        except nx.NetworkXNoPath:
            top_k_paths.append([target_seq_id, seq_id]) 
    
    return top_k_seqs, top_k_paths

def build_memory_tree(target_seq, retrieved_seqs, paths):
    tree = {
        'root': target_seq,
        'children': []
    }
    for seq, path in zip(retrieved_seqs, paths):
        tree['children'].append({
            'node': seq,
            'path': path,
            'leaves': [torch.tensor([len(p)])] 
        })
    return tree