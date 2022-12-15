import numpy as np
import torch
import random
import math
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected
from sklearn.neighbors import kneighbors_graph

def generate_helix(alpha = 1, beta = 1, seed = 1234, n_samples = 3000, n_neighbours = 5, features = 'coordinates', standardize=True):
  random.seed(seed)
  theta = 8 * math.pi * np.random.beta(a = alpha,b = beta, size= n_samples)
  phi = 4 * math.pi * np.random.beta(a = alpha, b = beta, size = n_samples)
  n =  np.random.uniform(0,1,n_samples)
  #x=(1.2+0.1*np.cos(phi))*np.cos(theta)
  #y=(1.2+0.1*np.cos(phi))*np.sin(theta)
  theta = np.sort(theta)
  x = n*np.cos(theta)
  y = n*np.sin(theta)
  #z=0.1*np.sin(phi)+theta/np.pi
  z = theta/np.pi 
  X = np.vstack([np.array(x), np.array(y), np.array(z)]).T
  A = kneighbors_graph(X, n_neighbours, mode='distance', include_self=False)
  edge_index, edge_weights = from_scipy_sparse_matrix(A)
  edge_index, edge_weights = to_undirected(edge_index, edge_weights)
  if standardize:
      preproc = StandardScaler()
      X = preproc.fit_transform(X)
  if features == 'coordinates':
      new_data = Data(x=torch.from_numpy(X).float(),
                      edge_index=edge_index,
                      edge_weight=edge_weights)
  else:
      new_data = Data(x=torch.eye(n_samples), edge_index=edge_index,
                      edge_weight=edge_weights)
  return x,y,z, A, new_data

#' Toroidal Helix by Coifman & Lafon
#' @param N integer; Number of datapoints.
#' @param sample_rate numeric; Sampling rate.
#' @export
def toroidal_helix(N, sample_rate = 1.0):
  # source: https://rdrr.io/github/kcf-jackson/maniTools/src/R/simulate_data.R
  """
  input: number of samples
  output: data, color
  """
  noiseSigma = 0.05 #noise parameter
  t = np.arange(1,N+1) / N
  t = (t**(sample_rate)) * 2 * np.pi
  noise = noiseSigma * np.random.normal(size = (N,3))
  x =  (2+np.cos(8*t))*np.cos(t) + noiseSigma * np.random.normal(size = N)
  y =  (2+np.cos(8*t))*np.sin(t) + noiseSigma * np.random.normal(size = N) 
  z = np.sin(8*t) +  noiseSigma * np.random.normal(size = N)
  return x,y,z, t


def get_vhelix(uniform=True, a = 0.5, b = 0.5, seed = 1234, n_samples = 3000, n_neighbours = 5, features = 'coordinates', standardize=True):
    # reference: https://github.com/goroda/diffusion_maps/blob/main/gendata.py
    # plot vertical helix
    npts = n_samples
    if uniform is True:
        time = np.linspace(0, 1, npts);
    else:
        np.random.seed(seed)
        time = np.sort(np.random.beta(a, b, size=(npts)))


    height = np.sin(1.0 * np.pi * time)
    radius = np.sin(2.0 * 2.0 * np.pi * time) + 2.0
    x = radius * np.cos(5 * 2 * np.pi * time)
    y = radius * np.sin(5 * 2 * np.pi * time)

    X = np.vstack([np.array(x), np.array(y), np.array(height)]).T
    A = kneighbors_graph(X, n_neighbours, mode='distance', include_self=False)
    edge_index, edge_weights = from_scipy_sparse_matrix(A)
    edge_index, edge_weights = to_undirected(edge_index, edge_weights)
    if standardize:
      preproc = StandardScaler()
      X = preproc.fit_transform(X)
    if features == 'coordinates':
        new_data = Data(x=torch.from_numpy(X).float(),
                        edge_index=edge_index,
                        edge_weight=edge_weights)
    else:
        new_data = Data(x=torch.eye(n_samples), edge_index=edge_index,
                        edge_weight=edge_weights)
    return x,y, height, time, A, new_data