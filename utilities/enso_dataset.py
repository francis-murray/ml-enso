# recovered from 071824 backup
import torch
from torch_geometric.data import Data, Dataset


class ENSOGraphDataset(Dataset):
    def __init__(self, feature_data, target_data, edge_index, pos):
        print(f"• ENSOGraphDataset feature_data.shape: {feature_data.shape}")
        self.feature_data = feature_data  # List of node feature matrices (one per time point)
        self.target_data = target_data  # List of E or C indices (one per time point)
        self.edge_index = edge_index  # Static edge index for all graphs
        self.num_graphs = len(feature_data)
        self.pos = pos
    
    def __len__(self):
        return self.num_graphs
    
    def __getitem__(self, idx):
        # print(f"__getitem__ feature_data: ", self.feature_data.shape)
        # print(f"__getitem__ target_data: ", self.target_data.shape)
        # allows to split the dataset using slicing
        if isinstance(idx, slice):
            # rearrange the dimensions from (num_time_steps, num_features, num_nodes) to
            # (num_time_steps, num_nodes, num_features) for slices of data.
            # sliced_feature_data = self.feature_data[idx].permute(0, 2, 1) # Transpose each slice
            return self.__class__(self.feature_data[idx], self.target_data.iloc[idx], self.edge_index, self.pos)

        
        # Transpose to shape (num_nodes, num_features)
        x = self.feature_data[idx].transpose(1, 0)  # Node features at time point idx
        y = self.target_data.iloc[idx]  # ENSO index at time point idx


        # Debugging information
        # print(f"            • ENSOGraphDataset __getitem__ method: Feature data x size: {x.size()}")
        # print(f"            • ENSOGraphDataset __getitem__ method: Edge index size: {self.edge_index.size()}")

        return Data(x=x, edge_index=self.edge_index, y=torch.tensor([y], dtype=torch.float), pos=self.pos)
