# util
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

def local_edge_encoder(h_i, h_j, alpha):
 
    return alpha*h_i + (1-alpha)*h_j

def macro_edge_encoder(raw_edges, adj_norm, gamma, W_list, q):

    X0 = raw_edges
    X = X0
    Zs = []
    for W in W_list:
        X = (1-gamma)*(adj_norm @ X) + gamma*X0
        Zs.append(F.relu(X @ W))
    # attention over scales
    scores = torch.stack([ (torch.tanh(Zs[l]@W_list[l] + Zs[0]@W_list[0]) * q).sum(dim=1)
                           for l in range(len(Zs)) ], dim=1)  # [E, L]
    alpha = F.softmax(scores, dim=1)  # [E, L]
    Z = sum(alpha[:,l].unsqueeze(1) * Zs[l] for l in range(len(Zs)))
    return Z

def entropy_sim(z_a, z_b):
  
    S = (z_a @ z_b.T).cpu().numpy()
    r, c = linear_sum_assignment(-S)
    vals = S[r, c]
    p = vals / (vals.sum() + 1e-8)
    h = -np.sum(p*np.log(p+1e-8))/np.log(len(p))
    return vals.mean() * (1+h)

def retrieve_topk(seq_embs, k):
   
    N = len(seq_embs)
    ret = {}
    for i in range(N):
        sims = [(j, entropy_sim(seq_embs[i], seq_embs[j]))
                for j in range(N) if j!=i]
        sims.sort(key=lambda x: x[1], reverse=True)
        ret[i] = [j for j,_ in sims[:k]]
    return ret

def construct_seq_labels(node_labels, graphseq):
 
    y = []
    for seq in graphseq:
        ints = list(map(int, seq))
        y.append(1.0 if any(node_labels[i]==1 for i in ints) else 0.0)
    return torch.tensor(y)

def seq_to_node_scores(seq_scores, seq_per_node):
   
    arr = np.array(seq_scores).reshape(-1, seq_per_node)
    return arr.mean(axis=1)
