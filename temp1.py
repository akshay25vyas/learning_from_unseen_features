# plot feature importance manually
import numpy as np
from sklearn.linear_model import LinearRegression
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import random

def initialize(xs,ys,xt,zt,k):
    # for every xs find set of k nearest neighbours in xt and pick any one
    # here taking minimum and its corresponding zt as z^s
    m,d=xs.shape
    X=np.zeros((m,d+1),'float')
    for i in range(len(xs)):
        x=xs[i,:]
        mini=float('inf')
        for j in range(len(xt)):
            xdash=xt[j,:]
            diff=np.subtract(x,xdash)
            pr=diff*diff
            val=np.sum(pr)
            if(val<mini):
                mini=val
                minin=j
        zdashs=zt[minin]
        X[i,0:7]=x
        X[i,7]=zdashs
    model = LinearRegression()
    model.fit(X, ys)
    thetha_initial = model.coef_
    return thetha_initial

def make_source(z_t_index):
    # load data
    dataset = loadtxt('house_files/train.txt')
    # split data into X and y
    X,ys = dataset[:,1:9],dataset[:,0]
    xs=np.delete(X, z_t_index, 1)  # delete  column of X
    return xs,ys

def make_target(numb):
    ## we are finding the features with most importance value and then will call it z_t
    # load data
    dataset = loadtxt('house_files/test.txt')
    # split data into X and y
    X,yt = dataset[:,1:9],dataset[:,0]
    # fit model no training data
    model = XGBClassifier()
    #print model
    model.fit(X, yt)
    plot_importance(model)
    # feature importance select no. of features in zt according to numb , taken 1 i.e. argmax only
    z_t_index=np.argmax(model.feature_importances_)
    # making zt,xt
    zt=dataset[:,z_t_index]
        xt=np.delete(X, z_t_index, 1)  # delete  column of X
    return zt,xt,yt,z_t_index
#def upda_nn_target(xs,xt,zt,ys,thetha,ld):
#    np.linalg.norm(x.T)

def main_func():
    zt,xt,yt,z_t_index=make_target(1) # 1 represents how many featuers to kept in zt
    xs,ys=make_source(z_t_index) # making Source and target data
    k,gamma,ld=5,0.1,0.1 # number of neighbours, weight parameter,regularization parameter
    thetha_initial=initialize(xs,ys,xt,zt,k)
    print thetha_initial
    '''
    T=8 # no. of times you want to run iteration for updating thetha
    thetha=thetha_initial
    for it in range(T):
        for s in range(len(xs)):
            V_t_k=upda_nn_target(thetha) # for every xs[s] update its nearest neighbours in target domain
        for t in range(len(xt)):
            V_s_k=upda_nn_source(thetha) # for every xt[t] update its nearest neighbours in source domain
        # now we have V_t_k & V_s_k ,update thetha
    print thetha
    '''
main_func()
