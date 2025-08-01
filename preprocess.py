import pickle
import numpy as np
import os
import torch
import torch.optim as optim
from deepwalk import SCRADSequenceConstructor 
from dataset import Dataset
from args import args
from util import construct_coherence_label, get_seq_with_edges, compute_dual_layer_edge_embeddings
from models.gcn import GCN

def load_model(model_path, model, optimizer=None):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    return model, optimizer

def process_data(dataname):
    data_path = f'preprocessed/{dataname}/'
    os.makedirs(data_path, exist_ok=True)

    # Load raw dataset
    dataset = Dataset(dataname)
    attr = dataset.attr
    adj = dataset.adj
    node_label = dataset.label
    edge_index = dataset.edge_index
    num_nodes = dataset.num_nodes

    seq_constructor = SCRADSequenceConstructor(num_nodes, edge_index, undirected=True)
    seq_num = args.seq_num  
    seq_length = args.seq_length
    sequences = seq_constructor.build_total_sequences(seq_num, seq_length)
    graphseq, graphseq_dict = seq_constructor.concate(sequences, num_nodes, seq_num, seq_length)


    seq_label = construct_coherence_label(node_label, graphseq)

    model = GCN(attr.size(1), args.hidden_channels, 2, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model, optimizer = load_model(f'checkpoints/{dataname}/gcn_model.pth', model, optimizer)
    with torch.no_grad():
        _, node_embeddings = model(attr, adj)


    edge_embeddings = compute_dual_layer_edge_embeddings(node_embeddings, edge_index, adj)


    seq_data, edge_data = get_seq_with_edges(node_embeddings, edge_embeddings, edge_index, graphseq)

    graph_data = {
        'attr': attr,
        'adj': adj,
        'node_label': node_label,
        'edge_index': edge_index,
        'node_embeddings': node_embeddings,
        'edge_embeddings': edge_embeddings
    }
    with open(os.path.join(data_path, 'graph_data.pkl'), 'wb') as f:
        pickle.dump(graph_data, f)

    graphseq_data = {
        'seq_data': seq_data,
        'edge_data': edge_data,
        'seq_label': seq_label
    }
    with open(os.path.join(data_path, 'graphseq_data.pkl'), 'wb') as f:
        pickle.dump(graphseq_data, f)

    mapping_data = {
        'graphseq_dict': graphseq_dict,
        'graphseq': graphseq
    }
    with open(os.path.join(data_path, 'mapping_data.pkl'), 'wb') as f:
        pickle.dump(mapping_data, f)

    node_indices = list(graphseq_dict.keys())
    label_1_indices = [i for i in node_indices if node_label[i] == 1]  
    label_0_indices = [i for i in node_indices if node_label[i] == 0]  

    np.random.seed(42)
    np.random.shuffle(label_1_indices)
    np.random.shuffle(label_0_indices)


    train_ratio = 0.6
    train_label_1 = label_1_indices[:int(len(label_1_indices)*train_ratio)]
    train_label_0 = label_0_indices[:int(len(label_0_indices)*train_ratio)]
    remaining_label_1 = label_1_indices[int(len(label_1_indices)*train_ratio):]
    remaining_label_0 = label_0_indices[int(len(label_0_indices)*train_ratio):]
    
    valid_label_1 = remaining_label_1[:int(len(remaining_label_1)*0.5)]
    valid_label_0 = remaining_label_0[:int(len(remaining_label_0)*0.5)]
    test_label_1 = remaining_label_1[int(len(remaining_label_1)*0.5):]
    test_label_0 = remaining_label_0[int(len(remaining_label_0)*0.5):]

    train_node_indices = train_label_1 + train_label_0
    test_node_indices = valid_label_1 + valid_label_0 + test_label_1 + test_label_0  

    np.random.seed(55)
    np.random.shuffle(train_node_indices)
    np.random.shuffle(test_node_indices)

    split_node_indices = {
        'train_node_indices': train_node_indices,
        'test_node_indices': test_node_indices
    }
    with open(os.path.join(data_path, 'split_node_indices.pkl'), 'wb') as f:
        pickle.dump(split_node_indices, f)


    attr_train = attr[train_node_indices]
    adj_train = adj[train_node_indices][:, train_node_indices]
    node_label_train = node_label[train_node_indices]
    row_indices, col_indices = torch.nonzero(adj_train, as_tuple=True)
    edge_index_train = torch.stack((row_indices, col_indices))


    train_constructor = SCRADSequenceConstructor(len(train_node_indices), edge_index_train, undirected=True)
    train_sequences = train_constructor.build_total_sequences(seq_num, seq_length)
    graphseq_train, _ = train_constructor.concate(train_sequences, len(train_node_indices), seq_num, seq_length)

    seq_label_train = construct_coherence_label(node_label_train, graphseq_train)
    with torch.no_grad():
        _, node_embeddings_train = model(attr_train, adj_train)
    edge_embeddings_train = compute_dual_layer_edge_embeddings(node_embeddings_train, edge_index_train, adj_train)
    seq_data_train, edge_data_train = get_seq_with_edges(node_embeddings_train, edge_embeddings_train, edge_index_train, graphseq_train)

  
    graphseq_data_train = {
        'seq_data': seq_data_train,
        'edge_data': edge_data_train,
        'seq_label': seq_label_train
    }
    with open(os.path.join(data_path, 'train_graphseq_data.pkl'), 'wb') as f:
        pickle.dump(graphseq_data_train, f)

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    process_data(args.dataname)