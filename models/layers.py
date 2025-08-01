import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class LocalEdgeEncoder(Module):
    def __init__(self, in_dim, hidden_dim, alpha=0.5):
        super(LocalEdgeEncoder, self).__init__()
        self.alpha = alpha
        self.W_forward = nn.Linear(in_dim, hidden_dim)
        self.W_backward = nn.Linear(in_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, edge_emb, node_to_edges, edge_indices):
        agg_forward = []
        agg_backward = []
        for i in edge_indices:
            u, v = edge_indices[i]  

            u_edges = [e for e in node_to_edges.get(u, []) if e != i]
            agg_u = torch.mean(edge_emb[u_edges], dim=0) if u_edges else torch.zeros_like(edge_emb[i])
            agg_forward.append(self.W_forward(agg_u))
            
            
            v_edges = [e for e in node_to_edges.get(v, []) if e != i]
            agg_v = torch.mean(edge_emb[v_edges], dim=0) if v_edges else torch.zeros_like(edge_emb[i])
            agg_backward.append(self.W_backward(agg_v))
        
        agg_forward = torch.stack(agg_forward)
        agg_backward = torch.stack(agg_backward)
        
       
        gate = torch.sigmoid(self.gate(agg_forward + agg_backward))
        return gate * agg_forward + (1 - gate) * agg_backward


class GlobalEdgeEncoder(Module):
    def __init__(self, in_dim, hidden_dim, num_scales=3):
        super(GlobalEdgeEncoder, self).__init__()
        self.num_scales = num_scales
        self.scale_layers = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim) for _ in range(num_scales)
        ])
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, edge_emb, adj, gamma=0.1):
        num_edges = edge_emb.size(0)
        A = adj  
        I = torch.eye(A.size(0), device=A.device)
        A_tilde = A + I  
        D_tilde = torch.diag(torch.sum(A_tilde, dim=1))
        D_inv_sqrt = torch.inverse(torch.sqrt(D_tilde))
        A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt 

       
        scales = []
        current = edge_emb.T
        for l in range(self.num_scales):
            current = (1 - gamma) * (A_hat @ current.T).T + gamma * edge_emb.T
            scale_emb = F.relu(self.scale_layers[l](current.T))  
            scales.append(scale_emb)

       
        attn_weights = F.softmax(torch.stack([self.attention(s) for s in scales]), dim=0)
        out = torch.sum(torch.stack([attn_weights[l] * scales[l] for l in range(self.num_scales)]), dim=0)
        return out