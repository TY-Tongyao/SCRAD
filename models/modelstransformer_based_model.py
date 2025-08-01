import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.layers import LocalEdgeEncoder, GlobalEdgeEncoder


class TreeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TreeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.ioux = nn.Linear(input_dim, 3 * hidden_dim)
        self.iouh = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.fx = nn.Linear(input_dim, hidden_dim)
        self.fh = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, children):
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        c = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

       
        for i in range(x.size(0)):
           
            iou = self.ioux(x[i]) + self.iouh(h[i])
            i, o, u = torch.split(iou, self.hidden_dim, dim=0)
            i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

           
            f = torch.sigmoid(self.fx(x[i]) + self.fh(h[children[i]])) if children[i] else 0.0
            c[i] = i * u + torch.sum(f * c[children[i]], dim=0) if children[i] else i * u
            h[i] = o * torch.tanh(c[i])

        return h[0]  #


class SCRADModel(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_scales=3, k=5, llm_model='deepseek'):
        super(SCRADModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.k = k

        
        self.local_encoder = LocalEdgeEncoder(edge_feat_dim, hidden_dim)
        self.global_encoder = GlobalEdgeEncoder(edge_feat_dim, hidden_dim, num_scales)
        self.edge_fusion = nn.Linear(2 * hidden_dim, hidden_dim)

        
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)

       
        self.tree_lstm = TreeLSTM(hidden_dim, hidden_dim)

      
        self.llm_tokenizer = AutoTokenizer.from_pretrained(f"deepseek-ai/deepseek-coder-6b-base" if llm_model == 'deepseek' else "meta-llama/Llama-2-7b-hf")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            f"deepseek-ai/deepseek-coder-6b-base" if llm_model == 'deepseek' else "meta-llama/Llama-2-7b-hf",
            device_map="auto",
            load_in_4bit=True if llm_model == 'deepseek' else False
        )
        self.llm_model.eval()
        for param in self.llm_model.parameters():
            param.requires_grad = False

       
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, seq_data, edge_data, memory_trees):
        batch_size, seq_len, _ = seq_data.size()
        device = seq_data.device

     
        node_proj = self.node_proj(seq_data)  

    
        num_edges = edge_data.size(1)
        edge_indices = [(i, i+1) for i in range(num_edges)] 
        node_to_edges = {i: [e for e, (u, v) in enumerate(edge_indices) if u == i or v == i] for i in range(seq_len)}

        local_edge_emb = self.local_encoder(edge_data, node_to_edges, edge_indices)  
        global_edge_emb = self.global_encoder(edge_data, self._build_seq_adj(seq_len, device), gamma=0.1)  
        
       
        edge_emb = self.edge_fusion(torch.cat([local_edge_emb, global_edge_emb], dim=2))  

        
        tree_embeddings = []
        for b in range(batch_size):
            tree = memory_trees[b]
          
            nodes = [tree['root']] + [child['node'] for child in tree['children']] + [p for child in tree['children'] for p in child['leaves']]
            children = [list(range(1, 1+self.k))] + [[] for _ in range(self.k)] + [[] for _ in range(self.k)] 
            tree_emb = self.tree_lstm(torch.stack(nodes), children)
            tree_embeddings.append(tree_emb)
        tree_emb = torch.stack(tree_embeddings)  

   
        seq_rep = torch.mean(node_proj, dim=1) + torch.mean(edge_emb, dim=1) + tree_emb  

       
        coherence_scores = self._llm_coherence_assessment(seq_data, edge_emb, tree_emb)  

        
        output = self.classifier(seq_rep) + coherence_scores
        return output

    def _build_seq_adj(self, seq_len, device):
        adj = torch.ones(seq_len, seq_len, device=device) - torch.eye(seq_len, device=device)
        return adj

    def _llm_coherence_assessment(self, node_seqs, edge_emb, tree_emb):
        batch_size = node_seqs.size(0)
        scores = []
        
        for b in range(batch_size):
            
            node_desc = f"Node sequence: {[f'node_{i}' for i in torch.argmax(node_seqs[b], dim=1).tolist()]}"
            edge_desc = f"Edge coherence: {torch.mean(edge_emb[b]).item():.3f}"
            tree_desc = f"Similar sequences: {tree_emb[b].norm().item():.3f}"
            
           
            prompt = f"""Assess if the following graph sequence is coherent (1=incoherent, 0=coherent).
{node_desc}
{edge_desc}
{tree_desc}
Output only a single number (0 or 1) without explanation."""
            
            inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(node_seqs.device)
            with torch.no_grad():
                outputs = self.llm_model.generate(**inputs, max_new_tokens=1)
            pred = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            scores.append(float(pred) if pred in ['0', '1'] else 0.5) 
        
        return torch.tensor(scores, device=node_seqs.device).view(-1, 1)