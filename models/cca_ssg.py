import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.baseline_models import LogReg, GCN, MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CCA_SSG(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, lambd, N, dropout_rate, gnn_type, alpha , beta, add_self_loops, use_mlp = False):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers, dropout_rate, gnn_type, alpha , beta, add_self_loops)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)
        self.lambd = lambd
        self.N = N

    def get_embedding(self, data):
        out = self.backbone(data.x, data.edge_index)
        return out.detach()

    def forward(self, data1, data2):
        h1 = self.backbone(data1.x, data1.edge_index)
        h2 = self.backbone(data2.x, data2.edge_index)
        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)
        return z1, z2

    def loss(self, z1, z2):
        c = torch.mm(z1.T, z2)
        c1 = torch.mm(z1.T, z1)
        c2 = torch.mm(z2.T, z2)
        c = c / self.N
        c1 = c1 / self.N
        c2 = c2 / self.N
        loss_inv = - torch.diagonal(c).sum()
        iden = torch.tensor(np.eye(c.shape[0])).to(device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()
        lambd = self.lambd
        ret = loss_inv + lambd * (loss_dec1 + loss_dec2)
        return ret