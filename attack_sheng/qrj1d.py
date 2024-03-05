# Generated with SMOP  0.41
# from libsmop import *
# qrj1d.m
import numpy as np

def qrj1d(X=None,varargin=[]):
    nargin = len(varargin) + 1
    
    X = X.copy()

    n,m = X.shape
    N = m // n
    BB = []
    #defaulat values
    ERR = 1e-6
    RBALANCE = 3
    ITER = 300
    ###
    MODE = 'B'

    if nargin == 0:
        print('you must enter the data')
        B = np.eye(n)
        return Y,B
    
    if nargin == 1:
        Err = ERR
        Rbalance = RBALANCE

    
    if nargin > 1:
        MODE = varargin[1].upper()

        if 'B' == MODE:
            ERR = varargin[2]
            mflag = 'D'

            if ERR >= 1:
                print('Error value should be much smaller than unity')
                B = []
                S = []
                return Y, B
        else:
            if 'E' == MODE:
                ERR = varargin[2]
                mflag = 'E'
                if ERR >= 1:
                    print('Error value should be much smaller than unity')
                    B=[]
                    S=[]
                    return Y, B
            else:
                if 'N' == MODE:
                    mflag = 'N'
                    ITER=varargin[2]
                    ERR=0
                    if ITER <= 1:
                        print('Number of itternations should be higher than \
one')
                        B=[]
                        S=[]
                        return Y, B
    
    if nargin == 4:
        RBALANCE=varargin[3]
        if ceil(RBALANCE) != logical_or(RBALANCE,RBALANCE) < 1:
            print('RBALANCE should be a positive \
integer')
            B=[]
            S=[]
            return Y, B
    
    JJ=[]
    EERR=[]
    EERRJ2=[]
    X1 = X.copy()
    B = np.eye(n,n)
    Binv = np.eye(n)
    J = 0

    for t in range(1,N+1):
        J = J + np.linalg.norm(X1[:,np.arange((t-1)*n+1-1, t*n)] - np.diag(np.diag(X[:, np.arange((t-1)*n+1-1,t*n)])),'fro') ** 2

    JJ.append(J)
    
#the following part implements a sweep 
##########################
    err = ERR * n + 1
    if MODE == 'B':
        ERR = ERR * n
    
    k=0
    while (err > ERR) and (k < ITER):

        k = k + 1
        L=np.eye(n)
        U=np.eye(n)
        Dinv=np.eye(n)
        

        for i in range(1, n):
            for j in range(0, i):
                      
                G=np.stack((- X[i,np.arange(i,m,n)] + X[j, np.arange(j,m,n)], -2 * X[i, np.arange(j,m,n)]), axis=0)
                
                U1,D1,V1=np.linalg.svd(np.dot(G,G.T))
                
                v=U1[:,0]
                tetha=1 / 2 * np.arctan(v[1] / v[0])
                c=np.cos(tetha)
                s=np.sin(tetha)
  
                h1=c*X[:,np.arange(j,m,n)]-s*X[:,np.arange(i,m,n)]
#
                h2=c*X[:,np.arange(i,m,n)]+s*X[:,np.arange(j,m,n)]

                X[:,np.arange(j,m,n)] = h1 

                X[:,np.arange(i,m,n)] = h2 

                h1=c*X[j,:] - s*X[i,:]

                h2=s*X[j,:] + c*X[i,:]

                X[j, :] = h1
                X[i, :] = h2

                h1 = c * U[j,:] - s * U[i,:]

                h2 = s * U[j,:] + c * U[i,:]

                U[j,:]=h1
                U[i,:]=h2
                
        for i in range(n):
            #for j=i+1:n
            rindex=[]
            Xj=[]
            for j in range(i + 1, n):
                
                cindex=np.arange(0,m)
                cindex = np.delete(cindex, np.arange(j,m,n))
                
                a = - (np.dot(X[i,cindex],X[j,cindex].T)) / (np.dot(X[i,cindex],X[i,cindex].T))
       
                if np.abs(a) > 1:
                    a=np.sign(a) * 1
                
                X[j,:]=a * X[i,:] + X[j,:];     

                I=np.arange(i,m,n)

                J=np.arange(j,m,n)
              
                X[:,J]=a * X[:,I] + X[:,J]

                L[j, :]=L[j, :] + a * L[i, :]
            
        B=np.dot(np.dot(L,U),B)
    
        err=np.max(np.max(np.abs(np.dot(L,U) - np.eye(n))))
        EERR.append(err)
        

        if k % RBALANCE == 0:
            d=sum(np.abs(X.T))

            D=np.diag(1.0 / d * N)

            Dinv=np.diag(d * N)

            J=0
            for t in range(1,N+1):
                X[:,np.arange((t-1)*n+1-1,t*n)] = np.dot(np.dot(D, X[:,np.arange((t-1)*n+1-1,t*n)]),D)
            B=np.dot(D,B)
         
        J=0

        BB.append(B)

        Binv=np.linalg.inv(B)
        
        
        for t in range(1, N+1):
            J = J + np.linalg.norm(X1[:, np.arange((t-1)*n+1-1, t*n)] - np.dot(np.dot(Binv,np.diag(np.diag(X[:,np.arange((t-1)*n+1-1,t*n)]))),Binv.T),'fro') ** 2
        
        
        
        JJ.append(J)

        if MODE == 'E':
            err=abs(JJ(-1 - 1) - JJ(-1)) / JJ(-1 - 1)
            EERRJ2.append(err)

    Y=X.copy()
    S={'iterations':k, 'LUerror':EERR,'J2error':JJ,'J2RelativeError':EERRJ2}

    return Y, B, (S, BB) 
   

