import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.io as sc
import seaborn as sb

def select_centoriod(path,k):
    data = sc.loadmat('%s'%(path)) 
    m , n = data['X'].shape
    X=data['X']
    centroids=np.zeros((k,n))
    ind=np.random.randint(0,m,k)
    for i in range(k):
        centroids[i,:] = X[ind[i],:]
    return centroids

def find_closestCentroid(path,centroids):
    data = sc.loadmat('%s'%(path))
    X=data['X']
    m = X.shape[0]
    k=centroids.shape[0]
    idx=np.zeros(m)
    for i in range (m):
        min_dist=1000000000000
        for j in range(k):
            dist=np.sum((X[i,:]-centroids[j,:])**2)
            if min_dist>dist:
                min_dist=dist
                idx[i]=j
    return idx            
                
def compute_centroid(path,idx,k):
    data = sc.loadmat('%s'%(path))
    X=data['X']
    m , n = X.shape
    centroids=np.zeros((k,n))
    for i in range(k):
        indices=np.where(idx==i)
        centroids[i,:]=(np.sum(X[indices,:],axis=1)/len(indices[0])).ravel()
    return centroids     
    
def run_Kmeans(path, centroids, max_iters):
    data = sc.loadmat('%s'%(path))
    X=data['X']
    m , n = X.shape
    k=centroids.shape[0]
    idx=np.zeros(m)
    c=centroids
    for i in range( max_iters):
        idx = find_closestCentroid('C:\\1.PC\\Career\\ML\\codes\\Clustering\\Example\\ex7data2', c)
        c=compute_centroid('C:\\1.PC\\Career\\ML\\codes\\Clustering\\Example\\ex7data2', idx, k)
    return idx , c    
        
    
data = sc.loadmat('C:\\1.PC\\Career\\ML\\codes\\Clustering\\Example\\ex7data2') 
print (data['X'].shape)
X=data['X']
selected_centeoids=select_centoriod('C:\\1.PC\\Career\\ML\\codes\\Clustering\\Example\\ex7data2', 3)
print(selected_centeoids)
idx=find_closestCentroid('C:\\1.PC\\Career\\ML\\codes\\Clustering\\Example\\ex7data2',selected_centeoids)
#print(idx)
c=compute_centroid('C:\\1.PC\\Career\\ML\\codes\\Clustering\\Example\\ex7data2',idx,3)
print(c)

for x in range(10):
    idx,centroids=run_Kmeans('C:\\1.PC\\Career\\ML\\codes\\Clustering\\Example\\ex7data2',selected_centeoids,x)
    cluster1=X[np.where(idx==0)[0],:]
    cluster2=X[np.where(idx==1)[0],:]
    cluster3=X[np.where(idx==2)[0],:]
    fig,ax=plt.subplots(figsize=(9,6))
    ax.scatter(cluster1[:,0],cluster1[:,1],s=30,c='r',label='Cluster 1 ')
    ax.scatter(centroids[0,0],centroids[0,1],s=300,c='r')
    
    ax.scatter(cluster2[:,0],cluster2[:,1],s=30,c='y',label='Cluster 2 ')
    ax.scatter(centroids[1,0],centroids[1,1],s=300,c='y')
    
    ax.scatter(cluster3[:,0],cluster3[:,1],s=30,c='g',label='Cluster 3 ')
    ax.scatter(centroids[2,0],centroids[2,1],s=300,c='g')
    ax.legend()