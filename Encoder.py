import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from annoy import AnnoyIndex
import os

class LightGCN(nn.Module):
    def __init__(self, num_users, num_st_nodes, num_intents, embedding_dim, num_layers, edge_index, device):
        super(LightGCN, self).__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.device = device

        self.num_users = num_users
        self.num_st_nodes = num_st_nodes
        self.num_intents = num_intents
        self.total_num_nodes = num_users + num_st_nodes + num_intents

        # Initialize embeddings
        self.embedding = nn.Embedding(self.total_num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        # Build and normalize adjacency matrix
        self.edge_index = edge_index
        self.A_hat = self.build_normalized_adj(edge_index)

    def build_normalized_adj(self, edge_index):
        num_nodes = self.total_num_nodes
        row, col = edge_index
        data = torch.ones(len(row)).float().to(self.device)

        adj = torch.sparse_coo_tensor(
            torch.stack([row, col]),
            data,
            torch.Size([num_nodes, num_nodes])
        ).coalesce()

        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

        D_inv_sqrt = deg_inv_sqrt

        row_indices, col_indices = adj.indices()
        values = adj.values()
        norm_values = D_inv_sqrt[row_indices] * values * D_inv_sqrt[col_indices]

        A_hat = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices]),
            norm_values,
            adj.size()
        ).coalesce().to(self.device)

        return A_hat

    def forward(self):
        all_embeddings = self.embedding.weight
        embeddings_list = [all_embeddings]

        for layer in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.A_hat, all_embeddings)
            embeddings_list.append(all_embeddings)

        final_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)

        user_emb_final = final_embeddings[:self.num_users]
        st_emb_final = final_embeddings[self.num_users:self.num_users + self.num_st_nodes]
        intent_emb_final = final_embeddings[self.num_users + self.num_st_nodes:]
        return user_emb_final, st_emb_final, intent_emb_final

class BPRDataset(Dataset):
    def __init__(self, interactions, num_intents, user_intent_set):
        self.interactions = interactions
        self.num_intents = num_intents
        self.user_intent_set = user_intent_set

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user_idx, st_idx, pos_intent_idx = self.interactions[idx]
        while True:
            neg_intent_idx = np.random.randint(self.num_intents)
            if neg_intent_idx not in self.user_intent_set[user_idx]:
                break
        return user_idx, st_idx, pos_intent_idx, neg_intent_idx

def train_lightgcn(train_data_path, save_dir='./model_output/', epochs=20, embedding_dim=64, num_layers=3, batch_size=2048):
    """
    Train LightGCN model on the given training data
    
    Args:
        train_data_path: Path to training data CSV file
        save_dir: Directory to save model outputs
        epochs: Number of training epochs
        embedding_dim: Dimension of embeddings
        num_layers: Number of GCN layers
        batch_size: Training batch size
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Load training data
    train_data = pd.read_csv(train_data_path)

    # Build node mappings
    user_ids = train_data['userid'].unique()
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    num_users = len(user_id_map)

    order_intentions = train_data['order_intention'].unique()
    intent_id_map = {intent: idx for idx, intent in enumerate(order_intentions)}
    num_intents = len(intent_id_map)

    hours = train_data['hour'].unique()
    loc_scenes = train_data['loc_scene'].unique()
    st_nodes = [f"{hour}_{loc_scene}" for hour in hours for loc_scene in loc_scenes]
    st_id_map = {st_node: idx for idx, st_node in enumerate(st_nodes)}
    num_st_nodes = len(st_id_map)

    # Build training interactions
    train_interactions = []
    user_intent_set = {}
    for idx, row in train_data.iterrows():
        user = row['userid']
        hour = row['hour']
        loc_scene = row['loc_scene']
        order_intention = row['order_intention']

        if user in user_id_map and order_intention in intent_id_map:
            user_idx = user_id_map[user]
            st_node = f"{hour}_{loc_scene}"
            if st_node not in st_id_map:
                st_id_map[st_node] = len(st_id_map)
            st_idx = st_id_map[st_node]
            intent_idx = intent_id_map[order_intention]

            train_interactions.append((user_idx, st_idx, intent_idx))
            user_intent_set.setdefault(user_idx, set()).add(intent_idx)

    # Build edge index
    edge_index = [[], []]
    for user_idx, st_idx, intent_idx in train_interactions:
        st_idx += num_users
        intent_idx += num_users + num_st_nodes

        # User-Intent edges
        edge_index[0].extend([user_idx, intent_idx])
        edge_index[1].extend([intent_idx, user_idx])

        # Spatiotemporal-Intent edges
        edge_index[0].extend([st_idx, intent_idx])
        edge_index[1].extend([intent_idx, st_idx])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = edge_index.to(device)
    model = LightGCN(num_users, num_st_nodes, num_intents, embedding_dim, num_layers, edge_index, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create dataset and dataloader
    dataset = BPRDataset(train_interactions, num_intents, user_intent_set)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            user_idx_batch, st_idx_batch, pos_intent_idx_batch, neg_intent_idx_batch = [b.to(device) for b in batch]

            user_emb, st_emb, intent_emb = model()

            u_e = user_emb[user_idx_batch]
            st_e = st_emb[st_idx_batch]
            pos_i_e = intent_emb[pos_intent_idx_batch]
            neg_i_e = intent_emb[neg_intent_idx_batch]

            # Calculate BPR loss
            x_ui = torch.sum((u_e + st_e) * pos_i_e, dim=1)
            x_uj = torch.sum((u_e + st_e) * neg_i_e, dim=1)
            loss = -torch.mean(torch.log(torch.sigmoid(x_ui - x_uj)))

            # Add regularization
            reg_loss = 1e-4 * (u_e.norm(2).pow(2) + st_e.norm(2).pow(2) +
                            pos_i_e.norm(2).pow(2) + neg_i_e.norm(2).pow(2)) / u_e.size(0)
            loss += reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

    # Save final embeddings and mappings
    model.eval()
    with torch.no_grad():
        user_emb_final, st_emb_final, intent_emb_final = model()
        
    user_embeddings = user_emb_final.cpu().numpy()
    st_embeddings = st_emb_final.cpu().numpy()
    
    np.save(f'{save_dir}/user_embeddings.npy', user_embeddings)
    np.save(f'{save_dir}/st_embeddings.npy', st_embeddings)
    
    with open(f'{save_dir}/user_id_map.pkl', 'wb') as f:
        pickle.dump(user_id_map, f)
    with open(f'{save_dir}/st_id_map.pkl', 'wb') as f:
        pickle.dump(st_id_map, f)
    with open(f'{save_dir}/intent_id_map.pkl', 'wb') as f:
        pickle.dump(intent_id_map, f)

    return model, user_embeddings, st_embeddings, user_id_map, st_id_map, intent_id_map

def build_annoy_index(train_df, user_embeddings, st_embeddings, user_id_map, st_id_map, save_dir='./model_output/'):
    """
    Build Annoy index for efficient similarity search
    """
    print("Building Annoy index for similarity search...")
    
    embedding_dim = user_embeddings.shape[1]
    train_node_embeddings = []
    annoy_index = AnnoyIndex(embedding_dim, 'angular')

    for idx, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
        userid = row['userid']
        hour = row['hour']
        loc_scene = row['loc_scene']

        # Get user embedding
        user_embedding = np.zeros(embedding_dim)
        if userid in user_id_map:
            user_idx = user_id_map[userid]
            user_embedding = user_embeddings[user_idx]
        else:
            user_embedding = np.random.randn(embedding_dim)

        # Get spatiotemporal embedding
        st_node = f"{hour}_{loc_scene}"
        st_embedding = np.zeros(embedding_dim)
        if st_node in st_id_map:
            st_idx = st_id_map[st_node]
            st_embedding = st_embeddings[st_idx]
        else:
            st_embedding = np.random.randn(embedding_dim)

        # Combine embeddings
        node_embedding = user_embedding + st_embedding
        train_node_embeddings.append(node_embedding)
        annoy_index.add_item(idx, node_embedding)

    train_node_embeddings = np.array(train_node_embeddings)
    
    # Build and save index
    annoy_index.build(10)
    annoy_index.save(f'{save_dir}/annoy_index.ann')
    np.save(f'{save_dir}/train_node_embeddings.npy', train_node_embeddings)
    
    return annoy_index, train_node_embeddings

if __name__ == "__main__":
    # Example usage
    train_data_path = "train_data.csv"
    save_dir = "./model_output"
    
    # Train model
    model, user_embeddings, st_embeddings, user_id_map, st_id_map, intent_id_map = train_lightgcn(
        train_data_path=train_data_path,
        save_dir=save_dir,
        epochs=20,
        embedding_dim=64
    )
    
    # Build Annoy index
    train_df = pd.read_csv(train_data_path)
    annoy_index, train_node_embeddings = build_annoy_index(
        train_df=train_df,
        user_embeddings=user_embeddings,
        st_embeddings=st_embeddings,
        user_id_map=user_id_map,
        st_id_map=st_id_map,
        save_dir=save_dir
    )
    
    print("Training and index building completed successfully!")
