import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pickle
from util import compute_auc
from torch_geometric.data import Data
from models.layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1), x 


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data.x, data.adj)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def validate(model, data):
    model.eval()
    with torch.no_grad():
        out, emb = model(data.x, data.adj)
        pred = out.argmax(dim=1)
        auc = compute_auc(pred[data.val_mask], data.y[data.val_mask])
        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        acc = int(correct) / int(data.val_mask.sum()) if data.val_mask.sum() > 0 else 0.0
    return acc, auc, emb


def test(model, data):
    model.eval()
    with torch.no_grad():
        out, emb = model(data.x, data.adj)
        pred = out.argmax(dim=1)
        auc = compute_auc(pred[data.test_mask], data.y[data.test_mask])
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum()) if data.test_mask.sum() > 0 else 0.0
    return acc, auc, emb


def load_model(model_path, model, optimizer=None):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    return model, optimizer


def main():
    from args import args
    import os

    
    save_dir = os.path.join(args.save_dir, args.dataname)
    os.makedirs(save_dir, exist_ok=True)

   
    from dataset import Dataset
    dataset = Dataset(args.dataname)

   
    x = dataset.attr.float()
    adj = dataset.adj.float()
    edge_index = dataset.edge_index.long()
    y = dataset.label.long()
    num_nodes = dataset.num_nodes

    
    split_path = f'preprocessed/{args.dataname}/split_node_indices.pkl'
    with open(split_path, 'rb') as f:
        splits = pickle.load(f)
    train_indices = splits['train_node_indices']
    test_indices = splits['test_node_indices']

    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_size = int(0.75 * len(train_indices))  
    train_mask[train_indices[:train_size]] = True
    val_mask[train_indices[train_size:]] = True  
    test_mask[test_indices] = True

    
    data = Data(
        x=x,
        adj=adj,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    
    model = GCN(
        nfeat=x.size(1),
        nhid=args.hidden_dim,
        nclass=2,
        dropout=args.dropout
    ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

   
    data = data.to(args.device)
    data.x = data.x.to(args.device)
    data.adj = data.adj.to(args.device)

   
    best_val_auc = 0.0
    for epoch in range(args.epochs):
        train_loss = train(model, data, optimizer, criterion)
        val_acc, val_auc, _ = validate(model, data)
        test_acc, test_auc, _ = test(model, data)

        print(f'Epoch: {epoch+1:03d}, Loss: {train_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, '
              f'Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(save_dir, 'gcn_model.pth'))

    print(f"Best GCN model saved to {save_dir}/gcn_model.pth")


if __name__ == '__main__':
    main()