import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, dropout=0.2, device='cuda'):
        super(Model, self).__init__()
        self.device = device
        output_dim = 256
        self.gnn = GNNModule(dropout, hidden_dim=384, output_dim=output_dim)

        self.fp_proj = nn.Sequential(
            nn.Linear(1363, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim)
        )

        self.regression_head = nn.Sequential(
            nn.Linear(output_dim*2,128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self.relu = nn.ReLU()

        self.fp_in_dim=1363


    def forward(self, batch_size, batch, fp_t):
        fp = fp_t
        if fp.dim() == 2 and fp.size(0) == 1 and fp.size(1) % self.fp_in_dim == 0:
            B = fp.size(1) // self.fp_in_dim
            fp = fp.view(B, self.fp_in_dim)
        elif fp.dim() == 1 and fp.numel() % self.fp_in_dim == 0:
            B = fp.numel() // self.fp_in_dim
            fp = fp.view(B, self.fp_in_dim)

        graph, _ = self.gnn(batch.x, batch.edge_index, batch.batch)
        fp = self.fp_proj(fp)
        emb = torch.cat([graph, fp], dim=-1)      

        prediction = self.regression_head(emb)

        return prediction

    def label_loss(self, pred, label):
        return F.smooth_l1_loss(pred, label, beta=0.5)

class GNNModule(nn.Module):
    def __init__(self, dropout=0.5, hidden_dim=128, output_dim=256):
        super(GNNModule, self).__init__()

        num_features_xd=84

        nn1 = nn.Sequential(
            nn.Linear(num_features_xd, hidden_dim), nn.ReLU(),nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim*2)
        )
        self.conv1 = GINConv(nn1)

        nn2 = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2), nn.ReLU(),nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, hidden_dim*2)
        )
        self.conv2 = GINConv(nn2)

        self.fc_g = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        x_g = self.relu(self.conv1(x, edge_index))
        x_g = self.relu(self.conv2(x_g, edge_index))

        x_mean = gap(x_g, batch)
        x_max  = gmp(x_g, batch)
        x_g    = torch.cat([x_mean, x_max], dim=1)

        x_g = self.fc_g(x_g)
        return x_g, 0



