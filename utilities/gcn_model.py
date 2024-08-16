import torch
from torch.nn import Linear, BatchNorm1d, Tanh, Dropout
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import torch.nn.functional as F


class GCN2LayerConcat(torch.nn.Module):
    def __init__(self, nb_input_features, nb_nodes_per_graph, hidden_dim1, hidden_dim2, output_dim, activation_func, dropout_rate=0, use_batch_norm=False):
        super().__init__()
        torch.manual_seed(1234)
        self.use_batch_norm = use_batch_norm

        self.conv1 = GCNConv(nb_input_features, hidden_dim1)
        self.bn1 = BatchNorm1d(hidden_dim1) if use_batch_norm else None
        self.act1 = activation_func
        self.dropout1 = Dropout(dropout_rate)

        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.bn2 = BatchNorm1d(hidden_dim2) if use_batch_norm else None
        self.act2 = activation_func
        self.dropout2 = Dropout(dropout_rate)

        self.fc = Linear(hidden_dim2 * nb_nodes_per_graph, output_dim)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        h = self.conv1(x, edge_index)
        if self.use_batch_norm:
            h = self.bn1(h)
        h = self.act1(h)
        h = self.dropout1(h)

        h = self.conv2(h, edge_index)
        if self.use_batch_norm:
            h = self.bn2(h)
        h = self.act2(h)  
        h = self.dropout2(h) # Final GNN embedding space

        # 2. Concatenate all node embeddings of each graph
        num_graphs = batch.max().item() + 1  # Determine the number of graphs in the batch
        h_concat = []
        for i in range(num_graphs):
            h_graph = h[batch == i]
            h_graph = h_graph.view(1, -1)  # Flatten to a single vector
            h_concat.append(h_graph)
        
        h_concat = torch.cat(h_concat, dim=0)  # Concatenate all graphs

        # 3. Apply a final linear regressor
        out = self.fc(h_concat)

        return out




# wrapper class for global_mean_pool
class GlobalMeanPool(torch.nn.Module):
    def __init__(self):
        super(GlobalMeanPool, self).__init__()

    def forward(self, x, batch):
        return global_mean_pool(x, batch)

    
    
class GCN2LayerMeanPool(torch.nn.Module):
    def __init__(self, nb_input_features, hidden_dim1, hidden_dim2, output_dim, dropout_rate=0, use_batch_norm=False):
        super().__init__()
        torch.manual_seed(1234)
        self.use_batch_norm = use_batch_norm

        self.conv1 = GCNConv(nb_input_features, hidden_dim1)
        self.bn1 = BatchNorm1d(hidden_dim1) if use_batch_norm else None
        self.tanh1 = Tanh()
        self.dropout1 = Dropout(dropout_rate)
        
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.bn2 = BatchNorm1d(hidden_dim2) if use_batch_norm else None
        self.tanh2 = Tanh()
        self.dropout2 = Dropout(dropout_rate)

        self.pool = GlobalMeanPool()
        self.fc = Linear(hidden_dim2, output_dim)

    def forward(self, x, edge_index, batch):

        # 1. Obtain node embeddings 
        h = self.conv1(x, edge_index)
        if self.use_batch_norm:
            h = self.bn1(h)
        h = self.tanh1(h)
        h = self.dropout1(h)

        h = self.conv2(h, edge_index)
        if self.use_batch_norm:
            h = self.bn2(h)
        h = self.tanh2(h)  
        h = self.dropout2(h) # Final GNN embedding space


        # 2. Readout layer
        # Pooling to get a single embedding for the entire graph
        # Returns batch-wise graph-level-outputs by averaging node features across the node dimension.
        h = self.pool(h, batch)

        # 3. Apply a final linear regressor
        out = self.fc(h)
        
        return out
    



class GCN3LayerConcat(torch.nn.Module):
    def __init__(self, nb_input_features, nb_nodes_per_graph, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(nb_input_features, hidden_dim1)
        self.tanh1 = Tanh()
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.tanh2 = Tanh()
        self.conv3 = GCNConv(hidden_dim2, hidden_dim3)
        self.tanh3 = Tanh()
        self.fc = Linear(hidden_dim3 * nb_nodes_per_graph, output_dim)

    def forward(self, x, edge_index, batch):

        # 1. Obtain node embeddings 
        h = self.conv1(x, edge_index)
        h = self.tanh1(h)
        h = self.conv2(h, edge_index)
        h = self.tanh2(h)
        h = self.conv3(h, edge_index)
        h = self.tanh3(h) # Final GNN embedding space

        # 2b. Alternative Concatenate all node embeddings of each graph
        num_graphs = batch.max().item() + 1  # Determine the number of graphs in the batch
        h_concat = []
        for i in range(num_graphs):
            h_graph = h[batch == i]
            h_graph = h_graph.view(1, -1)  # Flatten to a single vector
            h_concat.append(h_graph)
        
        h_concat = torch.cat(h_concat, dim=0)  # Concatenate all graphs

        # 3. Apply a final linear regressor
        out = self.fc(h_concat)

        return out



class GCN2LayerConcat2FCs(torch.nn.Module):
    def __init__(self, nb_input_features, nb_nodes_per_graph, hidden_dim1, hidden_dim2, output_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(nb_input_features, hidden_dim1)
        self.tanh1 = Tanh()
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.tanh2 = Tanh()
        # fc1_out_dim = hidden_dim2 * nb_nodes_per_graph // 8
        fc1_out_dim = nb_nodes_per_graph
        print("fc1_out_dim: ", fc1_out_dim)
        self.fc1 = Linear(hidden_dim2 * nb_nodes_per_graph, fc1_out_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = Linear(fc1_out_dim, output_dim)

    def forward(self, x, edge_index, batch):

        # 1. Obtain node embeddings 
        h = self.conv1(x, edge_index)
        h = self.tanh1(h)
        h = self.conv2(h, edge_index)
        h = self.tanh2(h) # Final GNN embedding space

        # 2b. Alternative Concatenate all node embeddings of each graph
        num_graphs = batch.max().item() + 1  # Determine the number of graphs in the batch
        h_concat = []
        for i in range(num_graphs):
            h_graph = h[batch == i]
            h_graph = h_graph.view(1, -1)  # Flatten to a single vector
            h_concat.append(h_graph)
        
        h_concat = torch.cat(h_concat, dim=0)  # Concatenate all graphs

        # 3. Apply a fully connected layer and a final linear regressor
        z = self.fc1(h_concat)
        z = self.relu1(z)
        out = self.fc2(z)

        return out
        