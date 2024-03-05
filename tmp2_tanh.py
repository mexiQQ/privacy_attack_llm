import torch
import torch.nn.functional as F
from constants import BERT_CLS_TOKEN, BERT_SEP_TOKEN, BERT_PAD_TOKEN
import numpy as np
from scipy.spatial import distance
from b import (
    no_tenfact,
    matlab_eigs,
    matlab_eigs2
)
# from attack_sheng.no_tenfact import no_tenfact

def compute_grads(model, x_embeds, y_labels, create_graph=False, return_pooler=False, return_first_token_tensor=False, cheat=False, debug=False, args=None):
    outs, ori_pooler_dense_input = model(
        inputs_embeds=x_embeds, 
        labels=y_labels, 
        return_first_token_tensor=True)
    gradients = torch.autograd.grad(outs.loss, model.parameters(), create_graph=create_graph, allow_unused=True)

    if not return_pooler:
        if return_first_token_tensor:
            return gradients, ori_pooler_dense_input
        else:
            return gradients
        
    sub_dimension = args.rd  
    m = args.hd - 768 
    d = sub_dimension
    B = len(x_embeds)
    
    # activation
    # even => relu/leakey relu => 特殊处理
    # odd => sigmoid/tanh => 特殊处理 
    # squer + cubic => no 特殊处理
    
    # second layer gradient
    g = gradients[-2].cpu().numpy()[1][768:].reshape(m)
    # first layer gradient
    g_hat = gradients[-4].cpu().numpy()[768:, :d]
    W = model.bert.pooler.dense.weight.data[768:, :d].cpu().numpy() # W => m x d

    # T1 = np.zeros((d, d, d))
    # T2 = np.zeros((d, d, d))
    # T3 = np.zeros((d, d, d))
    # for i in range(0, d):
    #     for j in range(0, d):
    #         for k in range(0, d):
    #             T1[i, j, k] = np.sum(g_hat[:, i] * W[:, j] * W[:, k])
    #             if j==k:
    #                 T1[i,j,k]=T1[i,j,k]-np.sum(g_hat[:, i])

    # for i in range(0,d):
    #     for j in range(0,d):
    #         for k in range(0,d):
    #             T2[i,j,k]=T1[j,k,i]
    #             T3[i,j,k]=T1[k,i,j]

    # T = (T1 + T2 + T3)/3
    # M = np.sum(T, 2)/np.sqrt(d)
    # M = M/m

    M = np.zeros((d, d))
    A = np.ones(d)
    A = A / np.linalg.norm(A)
    sum_ak_wk = np.dot(W, A) # m x d, d
    for i in range(0, d):
        for j in range(0, d):
            M[i, j] = np.sum((g_hat[:, i] * W[:, j] + g_hat[:,j] * W[:,i]) * sum_ak_wk)
    M=M/m


    V, D = matlab_eigs(M, B)
    # P_M = M @ np.linalg.pinv(M.T @ M) @ M.T
    # P_V = V @ np.linalg.inv(V.T @ V) @ V.T

    # WV = W @ V # m x B
    # gV = g_hat @ V # m X B  =>  m x d, d x B

    V = torch.from_numpy(V).float().cuda()

    return gradients, V 

   

