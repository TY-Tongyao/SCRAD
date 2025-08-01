import os.path
import pickle
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
from sklearn.metrics import roc_auc_score
from datetime import datetime
from models.scrad_model import SCRADModel  
from args import args
from util import (
    evaluation, compute_ascore_majority, get_sequence_indices,
    retrieve_top_k_sequences, build_memory_tree
)

def custom_collate_fn(batch):
    seq_data, edge_data, label = zip(*batch)
    return (torch.stack(seq_data), torch.stack(edge_data), 
            torch.tensor(label, dtype=torch.float).view(-1, 1))

class SCRADDataLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, mode, seq_length=None):
        super(SCRADDataLoader, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.seq_length = seq_length
        self.load_data()
 
        if mode == 'train':
            self.sequence_kg = self.build_sequence_kg()
            with open(os.path.join(data_path, 'sequence_kg.pkl'), 'wb') as f:
                pickle.dump(self.sequence_kg, f)
        else:
            with open(os.path.join(data_path, 'sequence_kg.pkl'), 'rb') as f:
                self.sequence_kg = pickle.load(f)

    def load_data(self):
    
        with open(self.data_path+'graphseq_data.pkl', 'rb') as f:
            data = pickle.load(f)
            self.seq_data = data['seq_data']  
            self.edge_data = data['edge_data']  
            self.seq_label = data['seq_label']  
        
       
        with open(self.data_path+'graph_data.pkl', 'rb') as f:
            data = pickle.load(f)
            self.node_embeddings = data['node_embeddings']
            self.edge_embeddings = data['edge_embeddings']
            self.node_label = data['node_label']
        
      
        with open(self.data_path+'mapping_data.pkl', 'rb') as f:
            data = pickle.load(f)
            self.graphseq_dict = data['graphseq_dict']
        
    
        with open(self.data_path+'split_node_indices.pkl', 'rb') as f:
            data = pickle.load(f)
            self.train_node_indices = data['train_node_indices']
            self.test_node_indices = data['test_node_indices']

    
        if self.mode == 'train':
            with open(self.data_path + 'train_graphseq_data.pkl', 'rb') as f:
                data = pickle.load(f)
                self.seq_data = data['seq_data']
                self.edge_data = data['edge_data']
                self.seq_label = data['seq_label']
            self.node_indices = self.train_node_indices
        elif self.mode in ['valid', 'test']:
            self.node_indices = self.test_node_indices
            seq_indices = get_sequence_indices(self.node_indices, self.graphseq_dict)
            self.seq_data = self.seq_data[seq_indices]
            self.edge_data = self.edge_data[seq_indices]
            self.seq_label = self.seq_label[seq_indices]
        
        self.node_label = self.node_label[self.node_indices]
        self.graphseq_dict = {i: self.graphseq_dict[i] for i in self.node_indices}

        print(f"===> {self.mode} data loaded successfully!")

    def build_sequence_kg(self):
        """Build sequence-derived knowledge graph"""
        kg = nx.Graph()
       
        for seq_idx, seq in enumerate(self.seq_data):
            kg.add_node(f"seq_{seq_idx}", type="sequence")
     
        seq_nodes = {}
        for seq_idx, seq in enumerate(self.seq_data):
            nodes_in_seq = set(torch.argmax(seq, dim=1).tolist())  
            seq_nodes[seq_idx] = nodes_in_seq
            for other_seq_idx in range(seq_idx):
                if seq_nodes[other_seq_idx].intersection(nodes_in_seq):
                    kg.add_edge(f"seq_{seq_idx}", f"seq_{other_seq_idx}")
        return kg

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, index):
        return self.seq_data[index], self.edge_data[index], self.seq_label[index]

def train_model(dataname, timestamp, model, valid_loader, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs=50, device='cuda'):
    model.to(device)
    best_valid_auc = 0.0
    checkpoints_dir = f"./checkpoints/{dataname}/"
    os.makedirs(checkpoints_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for seq_data, edge_data, labels in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            seq_data, edge_data, labels = seq_data.to(device), edge_data.to(device), labels.to(device)
            optimizer.zero_grad()
            
      
            retrieved_seqs, paths = [], []
            for seq in seq_data:
                top_k_seqs, top_k_paths = retrieve_top_k_sequences(
                    seq, train_dataloader.dataset.sequence_kg, 
                    train_dataloader.dataset.seq_data, k=5
                )
                retrieved_seqs.append(top_k_seqs)
                paths.append(top_k_paths)
            
         
            memory_trees = [build_memory_tree(seq, rs, p) for seq, rs, p in zip(seq_data, retrieved_seqs, paths)]
            
         
            outputs = model(seq_data, edge_data, memory_trees)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * seq_data.size(0)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    
        valid_auc = validate_model(model, valid_loader, valid_dataloader, criterion, device)

     
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            torch.save(model.state_dict(), f'./checkpoints/{dataname}/best_model{timestamp}.pth')
            logging.info(f'Saved best model with validation AUC: {best_valid_auc:.4f}')
            print(f'Saved best model with validation AUC: {best_valid_auc:.4f}')

  
    final_model_path = os.path.join(checkpoints_dir, f'final_model{timestamp}.pth')
    torch.save(model.state_dict(), final_model_path)
    logging.info(f'Saved final model at {final_model_path}')
    print(f'Saved final model at {final_model_path}')

def validate_model(model, valid_loader, valid_dataloader, criterion, device='cuda'):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for seq_data, edge_data, labels in tqdm(valid_dataloader, desc="Validating"):
            seq_data, edge_data, labels = seq_data.to(device), edge_data.to(device), labels.to(device)
            
      
            retrieved_seqs, paths = [], []
            for seq in seq_data:
                top_k_seqs, top_k_paths = retrieve_top_k_sequences(
                    seq, valid_loader.sequence_kg, 
                    valid_loader.seq_data, k=5
                )
                retrieved_seqs.append(top_k_seqs)
                paths.append(top_k_paths)
            
        
            memory_trees = [build_memory_tree(seq, rs, p) for seq, rs, p in zip(seq_data, retrieved_seqs, paths)]
            
          
            outputs = model(seq_data, edge_data, memory_trees)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * seq_data.size(0)

            all_labels.append(labels.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())


    epoch_loss = running_loss / len(valid_dataloader.dataset)
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    node_ascore = compute_ascore_majority(all_outputs, valid_loader.graphseq_dict)
    node_auc, f1, pr = evaluation(node_ascore, valid_loader.node_label)
    
    logging.info(f'Validation Loss: {epoch_loss:.4f}')
    logging.info(f'Node AUC: {node_auc:.4f}, PR: {pr:.4f}')
    print(f'Validation Loss: {epoch_loss:.4f}')
    print(f'Node AUC: {node_auc:.4f}, PR: {pr:.4f}')
    return node_auc

def test_model(dataname, timestamp, model, test_loader, test_dataloader, device='cuda'):
    model.load_state_dict(torch.load(f'checkpoints/{dataname}/best_model{timestamp}.pth'))
    model.to(device)
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for seq_data, edge_data, labels in tqdm(test_dataloader, desc="Testing"):
            seq_data, edge_data, labels = seq_data.to(device), edge_data.to(device), labels.to(device)
            
           
            retrieved_seqs, paths = [], []
            for seq in seq_data:
                top_k_seqs, top_k_paths = retrieve_top_k_sequences(
                    seq, test_loader.sequence_kg, 
                    test_loader.seq_data, k=5
                )
                retrieved_seqs.append(top_k_seqs)
                paths.append(top_k_paths)
            
        
            memory_trees = [build_memory_tree(seq, rs, p) for seq, rs, p in zip(seq_data, retrieved_seqs, paths)]
            
          
            outputs = model(seq_data, edge_data, memory_trees)
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

 
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    node_ascore = compute_ascore_majority(all_outputs, test_loader.graphseq_dict)
    node_auc, f1, pr = evaluation(node_ascore, test_loader.node_label)
    auc = roc_auc_score(all_labels, all_outputs)
    
    logging.info(f'Test AUC: {auc:.4f}')
    logging.info(f'Test Node AUC: {node_auc:.4f}, Node F1: {f1:.4f}, Node PR: {pr:.4f}')
    print(f'Test AUC: {auc:.4f}')
    print(f'Test Node AUC: {node_auc:.4f}, Node F1: {f1:.4f}, Node PR: {pr:.4f}')

if __name__ == "__main__":
    dataname = args.dataname
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    
    if args.mode == 'train':
        graphseq_data_path = f'./preprocessed/{dataname}/'
        train_loader = SCRADDataLoader(graphseq_data_path, 'train')
        valid_loader = SCRADDataLoader(graphseq_data_path, 'valid')
        train_dataloader = torch.utils.data.DataLoader(
            train_loader, batch_size=256, shuffle=True, num_workers=4, collate_fn=custom_collate_fn
        )
        valid_dataloader = torch.utils.data.DataLoader(
            valid_loader, batch_size=256, shuffle=False, num_workers=4, collate_fn=custom_collate_fn
        )

  
        model = SCRADModel(
            node_feat_dim=train_loader.node_embeddings.size(1),
            edge_feat_dim=train_loader.edge_embeddings.size(1),
            hidden_dim=args.hidden_dim,
            num_layers=3
        )
        criterion = nn.BCEWithLogitsLoss()

        lr = 1e-3 if dataname in ['Reddit', 'Elliptic'] else 1e-2
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_model(dataname, timestamp, model, valid_loader, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs=50, device='cuda')
    elif args.mode == 'test':
        graphseq_data_path = f'./preprocessed/{dataname}/'
        test_loader = SCRADDataLoader(graphseq_data_path, 'test')
        test_dataloader = torch.utils.data.DataLoader(
            test_loader, batch_size=256, shuffle=False, num_workers=4, collate_fn=custom_collate_fn
        )

        model = SCRADModel(
            node_feat_dim=test_loader.node_embeddings.size(1),
            edge_feat_dim=test_loader.edge_embeddings.size(1),
            hidden_dim=args.hidden_dim,
            num_layers=3
        )
        test_model(dataname, timestamp, model, test_loader, test_dataloader, device='cuda')