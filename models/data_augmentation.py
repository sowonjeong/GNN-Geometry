import torch
from copy import deepcopy
import numpy as np
from torch_geometric.data import Data

def random_aug(data, feat_drop_rate, edge_mask_rate):
    n_node = data.num_nodes
    edge_mask = mask_edge(data, edge_mask_rate)
    feat = drop_feature(data.x, feat_drop_rate)

    new_data = deepcopy(data)

    src = new_data.edge_index[0]
    dst = new_data.edge_index[1]
    wgt = new_data.edge_weight

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]
    nwgt = wgt[edge_mask]

    new_data.edge_index = torch.vstack([nsrc, ndst])
    new_data.x = feat.to(dtype=torch.float32)
    new_data.edge_weight = nwgt  #drop edge_weight too

    return new_data, edge_mask;

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def mask_edge(data, mask_prob):
    E = data.num_edges

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx # idx of edges that is not masked