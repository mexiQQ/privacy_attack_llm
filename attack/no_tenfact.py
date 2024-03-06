# Generated with SMOP  0.41
import numpy as np
from .qrj1d import qrj1d
from sktensor.core import ttm
from sktensor.dtensor import dtensor

                                                                               
def no_tenfact(T=None,L=None,k=None):
    
    p = T.shape[0]
    
    sweeps= [0, 0]
    # STAGE 1: Random projections
    
    M=np.zeros((p,int(p * L)))
    W=np.zeros((p,L))
                                       
    for l in range(L):
        
        W[:,l]=np.random.randn(p,)
        W[:,l]=W[:,l] / np.linalg.norm(W[:,l])
        M[:,np.arange((l-1+1)*p+1-1, (l+1)*p)]=np.squeeze(np.array(dtensor(T).ttm([dtensor(np.eye(p)),dtensor(np.eye(p)),dtensor(np.expand_dims(W[:,l], 1).T)])))
    
    D,U,S=qrj1d(M)
    
    # calculate the true eigenvalues across all matrices
    Ui=np.linalg.inv(U)
    Ui_norms=np.sqrt(np.sum(Ui ** 2, 0))
    Ui_normalized=1.0 / Ui_norms * Ui

    dot_products=np.dot(Ui_normalized.T,W)
    Lambdas=np.zeros((p,L))

    for l in range(L):
        Lambdas[:,l]=(np.diag(D[:,l*p:(l+1)*p]) / dot_products[:,l]) * (Ui_norms**2).T
        

    # calculate the best eigenvalues and eigenvectors
    
    idx0 = np.argsort(np.mean(np.abs(Lambdas),1))[::-1]  
    Lambda0 = np.mean(Lambdas[idx0[:k],:], 1)
    V=Ui_normalized[:,idx0[:k]]
    

    sweeps[0]=S[0]['iterations']
    sweeps[1]=S[0]['iterations']

    # STAGE 2: Plugin projections
    
    W = Ui_normalized
    M = np.zeros((p, p * W.shape[1]))
    

    for l in range(W.shape[1]):
        w=W[:,l]
        w=w / np.linalg.norm(w)
        M[:,(l)*p:(l+1)*p]=np.squeeze(np.array(dtensor(T).ttm([dtensor(np.eye(p)),dtensor(np.eye(p)),dtensor(np.expand_dims(w,1).T)])))
    
    
    D,U,S=qrj1d(M)
    

    Ui=np.linalg.inv(U)

    Ui_norm= 1.0 / np.sqrt(sum(Ui ** 2)) * Ui

    V1=Ui_norm

    sweeps[1]=sweeps[1] + S[0]['iterations']
    Lambda=np.zeros((p,))

    for l in range(p):
        Z=np.linalg.inv(V1)
        X=np.dot(np.dot(Z,M[:,l*p:(l+1)*p]),Z.T)
        Lambda= Lambda + np.abs(np.diag(X))
    
    
    
    idx=np.argsort(np.abs(Lambda))[::-1]

    V1=Ui_norm[:,idx[:k]]

    misc={}
    

    misc['V0'] = V

    misc['sweeps'] = sweeps 
    
    return V1,Lambda,misc
    