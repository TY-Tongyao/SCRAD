import os
import numpy as np
import torch
import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class Dataset:
    def __init__(self, name, prefix='datasets/'):
        self.name = name
        self.data_path = os.path.join(prefix, name)
        self.attr = None
        self.adj = None
        self.edge_index = None
        self.label = None
        self.num_nodes = None
        self.num_edges = None
        self.graph = None
        
        if name in ['Reddit', 'soc-redditHyperlinks']:
            self._load_reddit()
        elif name in ['Elliptic']:
            self._load_elliptic()
        elif name in ['Amazon', 'Musical_Instruments']:
            self._load_amazon()
        elif name in ['Question']:
            self._load_question()
        elif name in ['Heal-Fraud']:
            self._load_heal_fraud()
        elif name in ['Epinions']:
            self._load_epinions()
        else:
            raise ValueError(f"Dataset {name} not supported")

    def _load_reddit(self):
        """Load Reddit dataset (hyperlink network)"""
       
        edges_df = pd.read_csv(
            os.path.join(self.data_path, 'soc-redditHyperlinks-title.tsv'),
            sep='\t'
        )
        edges = edges_df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']].values
        
        
        nodes = np.unique(edges.flatten())
        node_id = {node: i for i, node in enumerate(nodes)}
        self.num_nodes = len(nodes)
        
       
        edge_list = [(node_id[u], node_id[v]) for u, v in edges]
        self.edge_index = torch.tensor(edge_list).T.contiguous()
        
        
        self.adj = torch.zeros((self.num_nodes, self.num_nodes))
        for u, v in edge_list:
            self.adj[u, v] = 1.0
            self.adj[v, u] = 1.0 
        
        
        self.attr = torch.eye(self.num_nodes)
        
      
        labels_df = pd.read_csv(os.path.join(self.data_path, 'reddit_labels.csv'))
        self.label = torch.zeros(self.num_nodes, dtype=torch.long)
        for _, row in labels_df.iterrows():
            if row['node'] in node_id:
                self.label[node_id[row['node']]] = row['label']

    def _load_elliptic(self):
      
        features_df = pd.read_csv(os.path.join(self.data_path, 'elliptic_txs_features.csv'), header=None)
        edges_df = pd.read_csv(os.path.join(self.data_path, 'elliptic_txs_edgelist.csv'))
        classes_df = pd.read_csv(os.path.join(self.data_path, 'elliptic_txs_classes.csv'))
     
        tx_ids = features_df[0].values
        tx_id = {tx: i for i, tx in enumerate(tx_ids)}
        self.num_nodes = len(tx_ids)
        
    
        self.attr = torch.tensor(features_df.iloc[:, 1:166].values, dtype=torch.float32)
        
   
        edges = edges_df.values
        edge_list = [(tx_id[u], tx_id[v]) for u, v in edges if u in tx_id and v in tx_id]
        self.edge_index = torch.tensor(edge_list).T.contiguous()
        
   
        self.adj = torch.zeros((self.num_nodes, self.num_nodes))
        for u, v in edge_list:
            self.adj[u, v] = 1.0
        
        
        self.label = torch.zeros(self.num_nodes, dtype=torch.long)
        for _, row in classes_df.iterrows():
            tx = row['txId']
            cls = row['class']
            if tx in tx_id and cls != 'unknown':
                self.label[tx_id[tx]] = 1 if cls == '1' else 0

    def _load_amazon(self):
        import json
        
    
        reviews = []
        with open(os.path.join(self.data_path, 'Musical_Instruments.json'), 'r') as f:
            for line in f:
                reviews.append(json.loads(line))
        
      
        users = list({r['reviewerID'] for r in reviews})
        items = list({r['asin'] for r in reviews})
        user_id = {u: i for i, u in enumerate(users)}
        item_id = {i: len(users) + j for j, i in enumerate(items)}
        self.num_nodes = len(users) + len(items)
        
      
        edge_list = [(user_id[r['reviewerID']], item_id[r['asin']]) for r in reviews]
        self.edge_index = torch.tensor(edge_list).T.contiguous()
        
      
        self.adj = torch.zeros((self.num_nodes, self.num_nodes))
        for u, v in edge_list:
            self.adj[u, v] = 1.0
            self.adj[v, u] = 1.0
        
      
        texts = [r['reviewText'] for r in reviews]
        tfidf = TfidfVectorizer(max_features=200)
        text_features = tfidf.fit_transform(texts).toarray()
        
      
        self.attr = torch.zeros((self.num_nodes, text_features.shape[1]))
        for i, r in enumerate(reviews):
            u = user_id[r['reviewerID']]
            v = item_id[r['asin']]
            self.attr[u] += torch.tensor(text_features[i], dtype=torch.float32)
            self.attr[v] += torch.tensor(text_features[i], dtype=torch.float32)
        
        
        self.attr /= torch.clamp(torch.sum(self.attr, dim=1, keepdim=True), min=1e-6)
        
        
        self.label = torch.zeros(self.num_nodes, dtype=torch.long)
      
        labels_df = pd.read_csv(os.path.join(self.data_path, 'amazon_labels.csv'))
        for _, row in labels_df.iterrows():
            if row['type'] == 'user' and row['id'] in user_id:
                self.label[user_id[row['id']]] = 1
            elif row['type'] == 'item' and row['id'] in item_id:
                self.label[item_id[row['id']]] = 1

    def _load_question(self):
      
        edges = np.loadtxt(os.path.join(self.data_path, 'edges.csv'), delimiter=',', skiprows=1)
        features = np.loadtxt(os.path.join(self.data_path, 'features.csv'), delimiter=',', skiprows=1)
        labels = np.loadtxt(os.path.join(self.data_path, 'labels.csv'), delimiter=',', skiprows=1)
        
        self.num_nodes = features.shape[0]
        self.attr = torch.tensor(features[:, 1:], dtype=torch.float32)  
        
        
        edge_list = edges[:, 1:3].astype(int)  # Skip index column
        self.edge_index = torch.tensor(edge_list).T.contiguous()
        
       
        self.adj = torch.zeros((self.num_nodes, self.num_nodes))
        for u, v in edge_list:
            self.adj[u, v] = 1.0
            self.adj[v, u] = 1.0
        
        
        self.label = torch.tensor(labels[:, 1], dtype=torch.long)  

    def _load_heal_fraud(self):
      
        claims = pd.read_csv(os.path.join(self.data_path, 'claims.csv'))
        providers = pd.read_csv(os.path.join(self.data_path, 'providers.csv'))
        
        
        provider_ids = providers['provider_id'].unique()
        node_id = {p: i for i, p in enumerate(provider_ids)}
        self.num_nodes = len(provider_ids)
        
        
        patient_providers = claims.groupby('patient_id')['provider_id'].unique()
        edge_list = []
        for _, providers in patient_providers.items():
            for i in range(len(providers)):
                for j in range(i+1, len(providers)):
                    u = node_id[providers[i]]
                    v = node_id[providers[j]]
                    edge_list.append((u, v))
        
        self.edge_index = torch.tensor(edge_list).T.contiguous()
        
        
        self.adj = torch.zeros((self.num_nodes, self.num_nodes))
        for u, v in edge_list:
            self.adj[u, v] += 1.0
            self.adj[v, u] += 1.0
        
       
        provider_features = providers.set_index('provider_id').iloc[:, 1:].values
        self.attr = torch.tensor(provider_features, dtype=torch.float32)
        
        
        self.label = torch.tensor(providers['fraud_label'].values, dtype=torch.long)

    def _load_epinions(self):
      
        with open(os.path.join(self.data_path, 'trust.txt'), 'r') as f:
            edges = [tuple(map(int, line.strip().split())) for line in f]
        
       
        nodes = np.unique([u for u, v in edges] + [v for u, v in edges])
        node_id = {n: i for i, n in enumerate(nodes)}
        self.num_nodes = len(nodes)
        
      
        edge_list = [(node_id[u], node_id[v]) for u, v in edges]
        self.edge_index = torch.tensor(edge_list).T.contiguous()
        
       
        self.adj = torch.zeros((self.num_nodes, self.num_nodes))
        for u, v in edge_list:
            self.adj[u, v] = 1.0
        
        
        self.attr = torch.eye(self.num_nodes)
        
        
        labels_df = pd.read_csv(os.path.join(self.data_path, 'epinions_labels.csv'))
        self.label = torch.zeros(self.num_nodes, dtype=torch.long)
        for _, row in labels_df.iterrows():
            if row['user_id'] in node_id:
                self.label[node_id[row['user_id']]] = row['is_anomalous']

    def get_node_attribute_matrix(self):
        return self.attr.numpy()