import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

class GNNModel(torch.nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels):
        super().__init__()
        self.model_type = model_type

        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=2)
            self.conv2 = GATConv(hidden_channels * 2, hidden_channels, heads=1)
        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        elif model_type == 'GIN':
            nn1 = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(),
                                      torch.nn.Linear(hidden_channels, hidden_channels))
            nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(),
                                      torch.nn.Linear(hidden_channels, hidden_channels))
            self.conv1 = GINConv(nn1)
            self.conv2 = GINConv(nn2)
        else:
            raise ValueError("Unknown model type")

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
