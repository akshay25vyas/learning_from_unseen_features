# intercept kaunsi
# pca nhi kia

# plot feature importance manually
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.decomposition import PCA
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

def load_dataset(filename):
    y = []
    x = []
    N = 100
    with open(filename,'r') as f:
        head = [next(f) for i in xrange(N)]
        for line in head:
            split = line.split()
            y.append(int(split[0]))
            del(split[0])
            xtemp = [float(item.split(':')[1]) for item in split]
            x.append(xtemp)
    y = np.array(y)
    x = np.array(x)
    return y,x
def normalize(x):
    x=copy.deepcopy((x-np.min(x))*1.0/(np.max(x)-np.min(x)))
    return x

def initialize(xs,ysf,xt,zt,k,ld):
    # for every xs find set of k nearest neighbours in xt and pick any one
    # here taking minimum and its corresponding zt as z^s
    m,d=xs.shape
    X=np.zeros((m,d+2),'float')
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
        X[i,0:255]=x
        X[i,255]=zdashs
        X[i,256]=1.0
    m1,d1=X.shape
    yuu=np.identity(d1+1)
    yuu[0,0]=0.0
    on=np.ones((m1,1))
    X=np.hstack([X,on])
    tomi=np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X),X)+ld*yuu),np.transpose(X)),ysf)
    return tomi

def make_source(z_t_index):
    # load data
    ys,X = load_dataset('usps.jpg')
    xs=np.delete(X, z_t_index, 1)  # delete  column of X
    return xs,ys

def make_target(numb):
    ## we are finding the features with most importance value and then will call it z_t
    # load data
    yt,X = load_dataset('usps.t')
    # fit model no training data
    model = XGBClassifier()
    #print model
    model.fit(X, yt)
    plot_importance(model)
    # feature importance select no. of features in zt according to numb , taken 1 i.e. argmax only
    z_t_index=np.argmax(model.feature_importances_)
    # making zt,xt
    zt=X[:,z_t_index]
    xt=np.delete(X, z_t_index, 1)  # delete  column of X
    return zt,xt,yt,z_t_index

def upda_nn_target(tomi,xsp,ysp,xt,zt,gamma,k):
    # making V_t_k
    m,d=xt.shape
    #print xt.shape
    X=np.zeros((m,d+2),'float')
    arr=np.zeros((m,1),'float')
    #ysp=np.reshape(10,1)
    #m1,d1,dw2=ysp.shape
    mat2=np.zeros((10,1),'float')
    for i in range(len(xt)):
        xdash=xt[i,:]
        X[i,0:255]=xdash
        X[i,255]=zt[i]
        X[i,256]=1.0
        diff=MahalanobisDist(xsp,xdash)
        pr=np.square(diff)
        d1=np.sum(pr)
        ycap=np.dot(tomi,X[i])
        mat2=np.multiply(ysp,ycap)
        d2=gamma*(1-np.sum(mat2))
        arr[i]=d2+d1
    sortedkeys = X[arr.argsort(axis=0)[::-1]]
    sortedkeys = np.reshape(sortedkeys, (m,d+2))
    return sortedkeys[0:k,:]

def upda_nn_source(tomi,xtp,ztp,xs,ysf,gamma,k):
    # making V_s_k
    m1,d1=ysf.shape
    mat2=np.zeros((m1,d1),'float')

    m,d=xs.shape
    X=np.zeros((1,d),'float')
    X=xtp
    X=np.append(X, ztp)
    X=np.append(X,1.0)
    #print X.shape
    ycap=np.dot(tomi,X)
    #print tomi.shape
    arr=np.zeros((m,1),'float')
    for i in range(len(xs)):
        xdash=xs[i,:]
        diff=MahalanobisDist(xtp,xdash)
        pr=np.square(diff)
        #pr=diff*diff
        #d1=np.linalg.norm(pr)
        d1=np.sum(pr)
        #d1=d1*d1
        mat2=np.multiply(ysf,ycap)
        d2=gamma*(1-np.sum(mat2))
        arr[i]=d1+d2
    sortedkeys=ysf[arr.argsort(axis=0)[::-1]]
    klo=np.zeros((k,10),'float')
    klo=np.reshape(sortedkeys[0:k,:],(k,10))
    return klo


def main_func():
    zt,xt,yt,z_t_index=make_target(1) # 1 represents how many featuers to kept in zt
    m,d=xt.shape
    zt=np.reshape(zt,(m,1))
    xs,ys=make_source(z_t_index) # making Source and target data
    k,gamma,ld=1,1,0.01 # number of neighbours, weight parameter,regularization parameter
    zt=normalize(zt)
    xt=normalize(xt)
    xs=normalize(xs)
    noc=10
    ysf=np.zeros((m,noc),'float')
    ytf=np.zeros((m,noc),'float')
    for i in range(m):
        ysf[i][int(ys[i]-1)]=1.0
    for i in range(m):
        ytf[i][int(yt[i]-1)]=1.0

    #tomi=initialize(xs,ysf,xt,zt,k,ld)

    tomi=np.zeros((10,257),'float')
    #intercept=0
    V_t_k=np.zeros((m,k,d+2),'float')
    V_s_k=np.zeros((m,k,10),'float')

    xf1=np.zeros((k*m,d+2),'float')
    xf2=np.zeros((k*m,d+2),'float')
    yf1=np.zeros((k*m,noc),'float')
    yf2=np.zeros((k*m,noc),'float')

    T=30 # no. of times you want to run iteration for updating thetha
    for it in range(30):
        #print it
        #print "iteration"
        for s in range(len(xs)):
            V_t_k[s]=upda_nn_target(tomi,xs[s],ysf[s,:],xt,zt,gamma,k) # for every xs[s] update its nearest neighbours in target domain
            for j in range(k):
                yf1[s*k+j]=ysf[s]
        #print "fjkhasdfjkhsd"
        for t in range(len(xt)):
            V_s_k[t]=upda_nn_source(tomi,xt[t],zt[t],xs,ysf,gamma,k) # for every xt[t] update its nearest neighbours in source domain
            for j in range(k):
                xf2[t*k+j,0:d]=xt[t]
                xf2[t*k+j,d]=zt[t]

        xf1=np.reshape(V_t_k,(m*k,d+2))
        yf2=np.reshape(V_s_k, (m*k,10))
        finalx=np.vstack([xf1,xf2])
        #print yf1.shape
        #print yf2.shape
        finaly=np.vstack([yf1,yf2])
        #print finalx.shape
        # now we have V_t_k & V_s_k ,update thetha
        m1,d1=finalx.shape
        yuu=np.identity(d1)
        yuu[0,0]=0.0
        on=np.ones((m1,1))
        #finalx=np.hstack([finalx,on])
        tomi=np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(finalx),finalx)+ld*yuu),np.transpose(finalx)),finaly)
        #tomi=np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(np.hstack([finalx,on])),np.hstack([finalx,on]))+ld*yuu),np.transpose(np.hstack([finalx,on]))),finaly)

        y_pred=np.dot(finalx,tomi)
        answer=np.sqrt(mean_squared_error(finaly, y_pred))
        tomi=np.transpose(tomi)
        #print tomi.shape
        print answer

    #Xtes=np.hstack([xt,zt])
    #y_pred=np.dot(thetha,Xtes.transpose())+intercept
    #answer=np.sqrt(mean_squared_error(yt, np.transpose(y_pred)))
    #print answer


main_func()
