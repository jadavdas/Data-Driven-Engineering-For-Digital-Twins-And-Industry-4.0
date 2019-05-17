#%%
test = "Hello World"
print("test: ", test)

#%%
import math

def basic_sigmoid(x):
    s = 1.0 / (1.0 + math.exp(-x))
    return s

result = basic_sigmoid(3)
print(result)

#%%
x = [1, 2, 3]
result = basic_sigmoid(x)
print(result)

#%%
import numpy as np
x = np.array([1, 2, 3])
result = np.exp(x)
print(result)
print(x + 3)

#%%
import numpy as np

def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s

x = np.array([1, 2, 3])
result = sigmoid(x)
print(result)

#%%
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s

x = np.linspace(-8, 8, 100)
result = sigmoid(x)
plt.plot(x,result,'r')

#%%
import numpy as np

def sigmoid_derivative(x):
    s = 1.0 / (1.0 + np.exp(-x))
    ds = s * (1 - s)
    return ds

x = np.array([1, 2, 3])
result = sigmoid_derivative(x)
print(result)

#%%
def image2vector(image):
    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
    return v

image = np.array([
                    [[0.67826139, 0.29380381],
                     [0.90714982, 0.52835647],
                     [0.4215251, 0.45017551]],
                     
                    [[0.92814219, 0.96677647],
                     [0.85304703, 0.52351845],
                     [0.19981397, 0.27417313]],
                     
                    [[0.60659855, 0.00533165],
                     [0.10820313, 0.49978937],
                     [0.34144279, 0.94630077]]
               
                   ])

result = image2vector(image)
print(str(result))

#%%
import numpy as np

def normalizeRows(x):
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    x = x/x_norm
    
    return x

x = np.array([
        [0, 3, 4],
        [1, 6, 4]
        ])
    
result = normalizeRows(x)
print(result)

#%%

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp/x_sum
    print("x_exp shape:", np.shape(x_exp))
    print("x_sum shape:", np.shape(x_sum))
    
    return s

x = np.array([
        [9, 2, 5, 0,0],
        [7, 5, 0,0, 0]
        ])
result = softmax(x)
print(result)

#%%
import time
import numpy as np

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i]*x2[i]
toc = time.process_time()
print("dot = ", str(dot) + "\n ---- Computation time = ", str(1000*(toc-tic)), "ms")

### CLASSIC OUTER PRODUCT IMPLWMWNTATION ###
tic = time.process_time()
outer = np.zeros((len(x1),len(x2)))
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i]*x2[j]
toc = time.process_time()
print("outer = ", str(outer) + "\n ---- Computation time = ", str(1000*(toc-tic)), "ms")

### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc = time.process_time()
print("elementwise multiplication = ", str(mul) + "\n ---- Computation time = ", str(1000*(toc-tic)), "ms")

### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
W = np.random.rand(3,len(x1))
tic = time.process_time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j]*x1[j]
toc = time.process_time()
print("gdot = ", str(gdot) + "\n ---- Computation time = ", str(1000*(toc-tic)), "ms")

#%%
import time
import numpy as np

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.process_time()
dot = np.dot(x1,x2)
toc = time.process_time()
print("dot = ", str(dot) + "\n ---- Computation time = ", str(1000*(toc-tic)), "ms")

### VECTORIZED OUTER PRODUCT OF VECTORS ###
tic = time.process_time()
outer = np.outer(x1,x2)
toc = time.process_time()
print("dot = ", str(outer) + "\n ---- Computation time = ", str(1000*(toc-tic)), "ms")


### VECTORIZED ELEMENTWISE MULTIPLICATION ###
tic = time.process_time()
mul = np.multiply(x1,x2)
toc = time.process_time()
print("dot = ", str(mul) + "\n ---- Computation time = ", str(1000*(toc-tic)), "ms")

### VECTORIZED GENERAL DOT PRODUCT ###
W = np.random.rand(3,len(x1))
tic = time.process_time()
gdot = np.dot(W,x1)
toc = time.process_time()
print("gdot = ", str(gdot) + "\n ---- Computation time = ", str(1000*(toc-tic)), "ms")


#%% L1 norm

import numpy as np

def L1(yhat,y):
    loss = np.sum(np.abs(yhat-y))
    return loss

yhat = np.array([0.9,0.2,0.1,0.4,0.9])
y = np.array([1,0,0,1,1])
print("L1 = ", str(L1(yhat,y)))

#%% L2 norm

import numpy as np

def L2(yhat,y):
    loss = np.sum((yhat-y)**2)
    return loss

yhat = np.array([0.9,0.2,0.1,0.4,0.9])
y = np.array([1,0,0,1,1])
print("L1 = ", str(L2(yhat,y)))

