import numpy as np
from scipy.linalg import eigh as largest_eigh
from .no_tenfact import no_tenfact
from heapq import nlargest
import torch
import torch.nn.functional as F
from collections import Counter
def tensor_feature(g,W,L,B,m,d):
    M = np.zeros((d, d))
    a = np.sum(g)
    
    for i in range(0, d):
        for j in range(0, d):
            M[i, j] = (np.sum(g[:, i] * W[:, j])+np.sum(g[:,j]*W[:,i]))/2
    M=M/m

    # largest absolute eigenvalues
    D1, V1 = largest_eigh(M)
    D2 = np.abs(D1).tolist()
    D_index = list(map(D2.index, nlargest(B, D2)))
    V = V1[:, D_index]

    return V
