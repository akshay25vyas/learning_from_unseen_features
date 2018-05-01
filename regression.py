# plot feature importance manually
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import numpy as np
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import random
import copy

def MahalanobisDist(x, y):
    covariance_xy = np.cov(x,y, rowvar=0)
    inv_covariance_xy = np.linalg.inv(covariance_xy)
    xy_mean = np.mean(x),np.mean(y)
    x_diff = np.array([x_i - xy_mean[0] for x_i in x])
    y_diff = np.array([y_i - xy_mean[1] for y_i in y])
    diff_xy = np.transpose([x_diff, y_diff])

    md = []
    for i in range(len(diff_xy)):
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_covariance_xy),diff_xy[i])))
    return md

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
            diff=MahalanobisDist(x,xdash)
            pr=np.square(diff)
            val=np.sum(pr)
            if(val<mini):
                mini=val
                minin=j
        zdashs=zt[minin]
        X[i,0:7]=x
        X[i,7]=zdashs
    regr = linear_model.LinearRegression()
    regr.fit(X, ys)
    y_pred = regr.predict(X)
    thetha_initial = regr.coef_
    intercept=regr.intercept_
    return thetha_initial,intercept

def normalize(x):
    x=copy.deepcopy((x-np.min(x))*1.0/(np.max(x)-np.min(x)))
    return x

def make_source(z_t_index):
    # load data
    dataset = loadtxt('train.txt')
    # split data into X and y
    X,ys = dataset[:,1:9],dataset[:,0]
    xs=np.delete(X, z_t_index, 1)  # delete  column of X
    return xs,ys

def make_target(numb):
    ## we are finding the features with most importance value and then will call it z_t
    # load data
    dataset = loadtxt('test.txt')
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

def upda_nn_target(thetha,intercept,xsp,ysp,xt,zt,gamma,k):
    # making V_t_k
    m,d=xt.shape
    X=np.zeros((m,d+1),'float')
    arr=np.zeros((m,1),'float')
    for i in range(len(xt)):
        xdash=xt[i,:]
        X[i,0:7]=xdash
        X[i,7]=zt[i]
        diff=MahalanobisDist(xsp,xdash)
        #print xdash
        #print xsp
        #print diff
        pr=np.square(diff)
        #d1=np.linalg.norm(pr)

        #d1=d1*d1
        d1=np.sum(pr)
        interim=np.dot(thetha,np.transpose(X[i]))
        ycap=interim+intercept
        d2=gamma*(ysp-ycap)*(ysp-ycap)
        arr[i]=d2+d1
    sortedkeys = X[arr.argsort(axis=0)[::-1]]
    sortedkeys = np.reshape(sortedkeys, (m,d+1))
    return sortedkeys[0:k,:]

def upda_nn_source(thetha,intercept,xtp,ztp,xs,ys,gamma,k):
    # making V_s_k
    m,d=xs.shape
    X=np.zeros((1,d),'float')
    X=xtp
    X=np.append(X, ztp)
    #print X.shape
    interim=np.dot(thetha,np.transpose(X))
    ycap=interim+intercept

    arr=np.zeros((m,1),'float')
    for i in range(len(xs)):
        xdash=xs[i,:]
        diff=MahalanobisDist(xtp,xdash)
        pr=np.square(diff)
        #pr=diff*diff
        #d1=np.linalg.norm(pr)
        d1=np.sum(pr)
        #d1=d1*d1
        d2=gamma*(ycap-ys[i])*(ycap-ys[i])
        arr[i]=d1+d2
    sortedkeys=ys[arr.argsort(axis=0)[::-1]]
    return sortedkeys[0:k,:]


def main_func():
    zt,xt,yt,z_t_index=make_target(1) # 1 represents how many featuers to kept in zt
    m,d=xt.shape
    zt=np.reshape(zt,(m,1))
    xs,ys=make_source(z_t_index) # making Source and target data
    zt=normalize(zt)
    xt=normalize(xt)
    xs=normalize(xs)

    k,gamma,ld=5,1,0.01 # number of neighbours, weight parameter,regularization parameter
    lear_rate=0.01
    thetha,intercept=initialize(xs,ys,xt,zt,k)
    #print thetha
    #print intercept

    #thetha=np.zeros((1,d+1),'float')
    #intercept=0

    V_t_k=np.zeros((m,k,d+1),'float')
    V_s_k=np.zeros((m,k,1),'float')

    xf1=np.zeros((k*m,d+1),'float')
    xf2=np.zeros((k*m,d+1),'float')
    yf1=np.zeros((k*m,1),'float')
    yf2=np.zeros((k*m,1),'float')
    tomi=np.zeros((d+2,1),'float')
    omi=np.zeros((d+2,1),'float')
    regg=np.zeros((d+2,d+2),'float')
    T=160 # no. of times you want to run iteration for updating thetha
    for it in range(T):
        for s in range(len(xs)):
            V_t_k[s]=upda_nn_target(thetha,intercept,xs[s],ys[s],xt,zt,gamma,k) # for every xs[s] update its nearest neighbours in target domain
            for j in range(k):
                yf1[s*k+j]=ys[s]
        for t in range(len(xt)):
            V_s_k[t]=upda_nn_source(thetha,intercept,xt[t],zt[t],xs,ys,gamma,k) # for every xt[t] update its nearest neighbours in source domain
            for j in range(k):
                xf2[t*k+j,0:d]=xt[t]
                xf2[t*k+j,d]=zt[t]

        xf1=np.reshape(V_t_k,(m*k,d+1))
        yf2=np.reshape(V_s_k, (m*k,1))
        finalx=np.vstack([xf1,xf2])
        finaly=np.vstack([yf1,yf2])
        m1,d1=finalx.shape
        yuu=np.identity(d1+1)
        yuu[0,0]=0.0
        on=np.ones((m1,1))
        finalx=np.hstack([finalx,on])
        tomi=np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(finalx),finalx)+ld*yuu),np.transpose(finalx)),finaly)

        #print finalx
        #print finalx.shape
        # now we have V_t_k & V_s_k ,update thetha
        #regr = linear_model.LinearRegression()
        #regr.fit(finalx, finaly)
        #print finalx.shape
        #thetha = regr.coef_
        #thetha= thetha-(thetha*lear_rate*ld)/m1
        #intercept=regr.intercept_-regr.intercept_*lear_rate*ld/m1

        #y_pred=np.dot(thetha,finalx.transpose())+intercept
        y_pred=np.dot(finalx,tomi)

        answer=np.sqrt(mean_squared_error(finaly, y_pred))
        omi=copy.deepcopy(tomi)

        thetha[0]=tomi[0]
        thetha[1]=tomi[1]
        thetha[2]=tomi[2]
        thetha[3]=tomi[3]
        thetha[4]=tomi[4]

        thetha[5]=tomi[5]
        thetha[6]=tomi[6]
        thetha[7]=tomi[7]

        intercept=tomi[8]

        print answer

    Xtes=np.hstack([xt,zt])
    y_pred=np.dot(thetha,Xtes.transpose())+intercept
    answer=np.sqrt(mean_squared_error(yt, np.transpose(y_pred)))
    print answer

main_func()
