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
            M[i, j] = (np.sum(g[:, i] * W[:, j])+np.sum(g[:,j] * W[:,i]))/2
    M=M/m

    # largest absolute eigenvalues
    D1, V1 = largest_eigh(M)
    D2 = np.abs(D1).tolist()
    D_index = list(map(D2.index, nlargest(B, D2)))
    V = V1[:, D_index]

    WV = np.matmul(W, V)  # m x B
    gV=np.matmul(g,V) # m X B

    # tensor
    T1 = np.zeros((B, B, B, B))
    T2 = np.zeros((B, B, B, B))
    T3 = np.zeros((B, B, B, B))
    T4 = np.zeros((B, B, B, B))
    for i in range(0, B):
        for j in range(0, B):
            for k in range(0, B):
                for l in range(0, B):
                    T1[i, j, k, l] = np.sum(gV[:, i] * WV[:, j] * WV[:, k] * WV[:, l])
                    if j==k:
                        T1[i,j,k,l]=T1[i,j,k,l]-np.sum(gV[:,i]*WV[:, l])
                    if j==l:
                        T1[i,j,k,l]=T1[i,j,k,l]-np.sum(gV[:,i]*WV[:, k])
                    if k==l:
                        T1[i,j,k,l]=T1[i,j,k,l]-np.sum(gV[:,i]*WV[:, j])
    for i in range(0,B):
        for j in range(0,B):
            for k in range(0,B):
                for l in range(0,B):
                    T2[i,j,k,l]=T1[j,i,k,l]
                    T3[i,j,k,l]=T1[k,i,j,l]
                    T4[i,j,k,l]=T1[l,i,j,k]
    T=(T1+T2+T3+T4)/4
    T = np.sum(T, 3) / np.sqrt(d)
    T=T/m

    rec_X, _, misc = no_tenfact(T, L, B)

    new_recX = np.matmul(V, rec_X)

    return new_recX, V


    # E = np.zeros((n, n, n))
    # for i in range(0, n):
    #     for j in range(0, n):
    #         for k in range(0, n):
    #             E[i, j, k] = np.sum(g * Xi * Xj * Xk)
    #             if i==j:
    #                 E[i,j,k]=E[i,j,k]-np.sum(g, Xk)
    #             if j==k:
    #                 E[i,j,k]=E[i,j,k]-np.sum(g * Xi)
    #             if i==k:
    #                 E[i,j,k]=E[i,j,k]-np.sum(g * Xj)