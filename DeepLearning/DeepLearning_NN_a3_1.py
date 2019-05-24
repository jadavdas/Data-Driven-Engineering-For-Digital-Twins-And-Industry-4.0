#%% import package
import numpy as np
import matplotlib.pyplot as plt

#%% Generating some data
N = 100 # number of point per class
D = 2 # dimentionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N) # radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
    
# Lets visualize the data:
plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

#%% Inilialize the parameters
W = 0.01*np.random.randn(D,K)
b = np.zeros((1,K))

#%% Compute the class scores
scores = np.dot(X,W) + b
