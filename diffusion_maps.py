import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh


def find_diffusion_matrix(X=None, epsilon = 10, alpha=1.0):
    """Function to find the diffusion matrix P
        
        Parameters:
        alpha - to be used for gaussian kernel function
        X - feature matrix as numpy array
        
        Returns:
        P_prime, P, Di, K, D_left
    """ 
    dists = euclidean_distances(X, X)
    K = np.exp(-dists**2 / epsilon)
    
    d = np.sum(K, axis=0)
    D = np.diag(d**(-alpha))
    L_alpha = D @ K @ D
    d_alpha = np.sum(L_alpha, axis = 0)
    D_alpha = np.diag(d_alpha**(-1))
    M = L_alpha @ D_alpha

    return L_alpha, D_alpha, K, M


def find_diffusion_map(M, D_alpha, n_eign=3,t = 3):
    """Function to find the diffusion coordinates in the diffusion space
        
        >Parameters:
        P_prime - Symmetrized version of Diffusion Matrix P
        D_left - D^{-1/2} matrix
        n_eigen - Number of eigen vectors to return. This is effectively 
                    the dimensions to keep in diffusion space.
        
        >Returns:
        Diffusion_map as np.array object
    """   
    eigenValues, eigenVectors = eigh(M)
    idx = eigenValues.argsort()[::-1]
    eigenValues = np.real(eigenValues[idx])
    eigenVectors = np.real(eigenVectors[:,idx])
    coordinate =   eigenVectors @ np.diag(eigenValues**t)
    
    return coordinate[:,:n_eign]

def apply_diffusions(data, alpha = 0.5, n_eign = 2, n_diff = 3,epsilon = 1e-2):
        P_alpha, D_alpha, K, M = find_diffusion_matrix(data, alpha=alpha, epsilon = epsilon)
        d_maps = find_diffusion_map(M, D_alpha, n_eign=n_eign, t = n_diff)
        return d_maps