from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import numpy as np
from numpy import loadtxt	
# n_samples, n_features = 10, 5
dataset = loadtxt('train.txt')
    # split data into X and y
X,ys = dataset[:,1:9],dataset[:,0]
# xs=np.delete(X, z_t_index, 1)  # delete  column of X
clf = KernelRidge(alpha=1.0)
clf.fit(X, ys)

dataset = loadtxt('test.txt')
# split data into X and y
Xt,yt = dataset[:,1:9],dataset[:,0]

Ypred =  clf.predict(Xt)
# print Ypred
rmse = np.sqrt(mean_squared_error(ys, Ypred))
print rmse
KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='linear',
            kernel_params=None)