# preprocess.py
"""
预处理模块：端到端生成序列、边编码、相似度检索并保存
"""

import os, pickle
import torch
import torch.optim as optim
from deepwalk import deepwalk_hy
from dataset import Dataset
from util import (
    construct_seq_labels,
    local_edge_encoder,
    macro_edge_encoder,
    retrieve_topk
)
from models.gcn import GCN

def process_data(dataname, args):
    base = f'preprocessed/{dataname}'
    os.makedirs(base, exist_ok=True)

    # 1) 加载图数据
    ds = Dataset(dataname)
    attr, adj, labels, edge_idx = ds.attr, ds.adj, ds.label, ds.edge_index

    # 2) GCN 预训练节点嵌入
    gcn = GCN(attr.size(1), args.hidden_channels, 2, dropout=args.dropout)
    ckpt = torch.load(f'checkpoints/{dataname}/gcn_model.pth')
    gcn.load_state_dict(ckpt['model_state_dict'])
    gcn.eval()
    with torch.no_grad():
        _, node_emb = gcn(attr, adj)  # [N, d]

    # 3) DeepWalk 生成序列
    walker = deepwalk_hy(ds.num_nodes, edge_idx, undirected=True)
    walks = walker.build_total_corpus(args.seq_num*2, args.seq_length//2+1, alpha=args.alpha)
    graphseq, seq_idx = walker.concate(walks, ds.num_nodes, args.seq_num, args.seq_length)

    # 4) 构建序列标签
    seq_labels = construct_seq_labels(labels.numpy(), graphseq)

    # 5) 边编码
    src, dst = edge_idx
    raw_edges = (node_emb[src] + node_emb[dst]) / 2  # [E, d]
    E = raw_edges.size(0)
    adj_norm = torch.eye(E)  # 简易示例，亦可用图拓扑构建
    W_list = [torch.randn(node_emb.size(1), node_emb.size(1)) for _ in args.scales]
    q = torch.randn(node_emb.size(1))
    macro_edges = macro_edge_encoder(raw_edges, adj_norm, args.gamma, W_list, q)

    # 6) 序列嵌入 & Top-k
    seq_embs = [ torch.stack([node_emb[int(n)] for n in seq], dim=0)
                 for seq in graphseq ]
    retrieve_map = retrieve_topk(seq_embs, args.topk)

    # 7) 保存全部
    with open(f'{base}/graph.pkl','wb') as f:
        pickle.dump({'attr':attr,'adj':adj,'labels':labels,'edge_idx':edge_idx}, f)
    with open(f'{base}/seq_map.pkl','wb') as f:
        pickle.dump({'graphseq':graphseq,'seq_idx':seq_idx}, f)
    torch.save({'raw_edges':raw_edges,'macro_edges':macro_edges}, f'{base}/edge_emb.pt')
    with open(f'{base}/seq_data.pkl','wb') as f:
        pickle.dump({'seq_embs':seq_embs,'retrieve_map':retrieve_map,'seq_labels':seq_labels}, f)

    print("Preprocessing finished.")
