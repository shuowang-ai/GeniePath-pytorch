import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_dim, 256, heads=4)
        self.lin1 = torch.nn.Linear(in_dim, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(
            4 * 256, out_dim, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, out_dim)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x