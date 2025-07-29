# main.py
"""
SCRAD 训练 & 测试主流程
- SeqDataset: 加载预处理数据
- train: 训练循环
- evaluate: 验证 & 测试
"""

import os, pickle
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.transformer_based_model import TransformerForSequenceClassification
from util import seq_to_node_scores

class SeqDataset(Dataset):
    def __init__(self, base, split):
        g = pickle.load(open(f'{base}/seq_data.pkl','rb'))
        m = pickle.load(open(f'{base}/seq_map.pkl','rb'))
        splits = pickle.load(open(f'{base}/split.pkl','rb'))
        nodes = splits[split]
        self.graphseq = m['graphseq']
        self.seq_idx = m['seq_idx']
        self.seq_embs = g['seq_embs']
        self.retrieve = g['retrieve_map']
        self.labels = g['seq_labels']
        # 收集该 split 下所有序列索引
        self.indices = [j for n in nodes for j in self.seq_idx[n]]

    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        sid = self.indices[i]
        return ( self.seq_embs[sid],
                 torch.tensor(self.retrieve[sid],dtype=torch.long),
                 torch.tensor(self.labels[sid],dtype=torch.float32) )

def collate_fn(batch):
    seqs, tops, labs = zip(*batch)
    return seqs, torch.stack(tops), torch.stack(labs)

def train(args):
    device = args.device
    base = f'preprocessed/{args.dataname}'
    os.makedirs(f'checkpoints/{args.dataname}', exist_ok=True)

    train_ds = SeqDataset(base,'train')
    val_ds   = SeqDataset(base,'test')
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    model = TransformerForSequenceClassification(
        node_dim=args.hidden_channels,
        edge_dim=args.hidden_channels,
        topk=args.topk
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss()

    best_val = 1e9
    for ep in range(args.epochs):
        model.train()
        tot = 0
        for seqs, tops, labs in tqdm(train_loader, desc=f"Epoch {ep+1}"):
            opt.zero_grad()
            out = model(seqs, tops)  # 内部含 memory-tree + LLM 召回
            loss = crit(out.squeeze(), labs.to(device))
            loss.backward(); opt.step()
            tot += loss.item()
        val_loss = evaluate(model, val_loader, crit, device)
        print(f"Epoch{ep+1} train={tot/len(train_loader):.4f} val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f'checkpoints/{args.dataname}/best.pth')

def evaluate(model, loader, crit, device):
    model.eval(); tot=0; cnt=0
    with torch.no_grad():
        for seqs, tops, labs in loader:
            out = model(seqs, tops)
            loss = crit(out.squeeze(), labs.to(device))
            tot += loss.item()*len(seqs); cnt += len(seqs)
    return tot/cnt

def test(args):
    device = args.device
    base = f'preprocessed/{args.dataname}'
    test_ds = SeqDataset(base,'test')
    loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    model = TransformerForSequenceClassification(
        node_dim=args.hidden_channels,
        edge_dim=args.hidden_channels,
        topk=args.topk
    ).to(device)
    model.load_state_dict(torch.load(f'checkpoints/{args.dataname}/best.pth'))

    preds, labs = [], []
    model.eval()
    with torch.no_grad():
        for seqs, tops, lab in loader:
            out = model(seqs, tops).sigmoid().cpu().numpy()
            preds.extend(out.flatten())
            labs.extend(lab.numpy())
    # 序列->节点级评分 & AUC
    node_scores = seq_to_node_scores(np.array(preds), args.seq_num)
    from sklearn.metrics import roc_auc_score
    print("Node AUC:", roc_auc_score(pickle.load(open(f'{base}/graph.pkl','rb'))['labels'].numpy(), node_scores))

if __name__ == "__main__":
    import args
    if args.args.mode=='train':
        train(args.args)
    else:
        test(args.args)
