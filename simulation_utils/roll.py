import numpy as np
import scipy as sc
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected


def make_roll(c=0.6, v=4, omega=12, n_samples = 2000, n_neighbours = 30,
              a = 2, b = 2, scale=0.5, plot=True, features=None,
              standardize=True):

    t = np.random.beta(a=b, b=b, size=n_samples)
    u = np.random.beta(a=2, b=2, size=n_samples) 
    vals = sc.stats.beta.pdf(t, a, b)
    x =4 *(v * t+c)* np.cos(omega  *t) + np.random.normal(scale=scale, size=n_samples)
    y =4 *(v * t+c) * np.sin(omega  *t) + np.random.normal(scale=scale, size=n_samples)
    z = 2 * np.ceil(np.max(x)) * (u-.5)

    X = np.vstack([np.array(x), np.array(y), np.array(z)]).T
    A = kneighbors_graph(X, n_neighbours, mode='distance', include_self=False) # edge weight is given by distance
    edge_index, edge_weights = from_scipy_sparse_matrix(A)
    edge_index, edge_weights = to_undirected(edge_index, edge_weights)
    M = torch.max(edge_weights)
    if standardize:
        preproc = StandardScaler()
        X = preproc.fit_transform(X)
    if features == 'coordinates':
        new_data = Data(x=torch.from_numpy(X).float(),
                        edge_index=edge_index,
                        edge_weight=edge_weights/M)
    else:
        new_data = Data(x=torch.eye(n_samples), edge_index=edge_index,
                        edge_weight=edge_weights/M)
    return(X, t, new_data, u)