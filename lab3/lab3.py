
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

# ## Jupyter notebooks
#
# In this lab, you can use Jupyter <https://jupyter.org/> to get a nice layout of your code and plots in one document. However, you may also use Python as usual, without Jupyter.
#
# If you have Python and pip, you can install Jupyter with `sudo pip install jupyter`. Otherwise you can follow the instruction on <http://jupyter.readthedocs.org/en/latest/install.html>.
#
# And that is everything you need! Now use a terminal to go into the folder with the provided lab files. Then run `jupyter notebook` to start a session in that folder. Click `lab3.ipynb` in the browser window that appeared to start this very notebook. You should click on the cells in order and either press `ctrl+enter` or `run cell` in the toolbar above to evaluate all the expressions.

# ## Import the libraries
#
# Check out `labfuns.py` if you are interested in the details.

import numpy as np
import sys
from scipy import misc
from imp import reload
from labfuns import *
from sklearn import decomposition
from matplotlib.colors import ColorConverter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ## Bayes classifier functions to implement
#
# The lab descriptions state what each function should do.

# Note that you do not need to handle the W argument for this part
# in: labels - N x 1 vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels,W=None):
    C = set(labels)
    prior = np.zeros(shape=len(C))
    for k in C:
        if W == None:
            Nk = float(len([x for x in labels if x == k]))
            prior[k] = Nk/len(labels)
        else:
            for idx, l in enumerate(labels):
                if l == k:
                    prior[k] += W[idx]

    return prior

# Note that you do not need to handle the W argument for this part
# in:      X - N x d matrix of N data points
#     labels - N x 1 vector of class labels
# out:    mu - C x d matrix of class means
#      sigma - d x d x C matrix of class covariances
def mlParams(X,labels,W=None):
    C = set(labels)
    d = len(X[0])

    mu = np.zeros(shape=(len(C), d))
    sigma = np.zeros(shape=(d, d, len(C)))

    for k in C:
        sumMu = 0
        for idx, x in enumerate(X):
            if labels[idx] == k:
                if W == None:
                    sumMu += x
                else:
                    sumMu += x*W[idx]

        Nk = 0
        if W == None:
            Nk = len([x for x in labels if x == k])
        else:
            Nk = sum([W[idx] for idx, x in enumerate(labels) if x == k])
        mu[k] = sumMu/Nk

        sumSigma = np.zeros(shape=(d, d))
        for idx, x in enumerate(X):
            if labels[idx] == k:
                matrix = x - mu[k]
                matrix = np.matrix(matrix)
                sumSigma += (matrix.transpose() * matrix)
                if W != None:
                    sumSigma *= W[idx]

        sigma[:,:,k] = sumSigma / Nk

    # sigma = np.array(sigma)
    return mu, sigma

# in:      X - N x d matrix of M data points
#      prior - C x 1 vector of class priors
#         mu - C x d matrix of class means
#      sigma - d x d x C matrix of class covariances
# out:     h - N x 1 class predictions for test points
def classify(X,prior,mu,sigma,covdiag=True):
    N = len(X)
    C = len(prior)
    d = len(X[0])
  
    h = np.zeros(shape=N)
    t1 = np.zeros(shape=(C))
    t3 = np.zeros(shape=(C))
    for k in range(C):
        t3[k] = np.log(prior[k])

    if not covdiag:
        L = np.zeros(shape=(C, d, d)) if d > 1 else np.zeros(shape=(C, 2, 2))

        for k in range(C):
            L[k] = np.linalg.cholesky(sigma[:,:,k])
            t1[k] = -1 * sum([np.log(L[k,i,i]) for i in range(d)])

        for idx, x in enumerate(X):
            lastMax = -sys.maxint - 1
            delta = np.zeros(shape=C)
            for k in range(C):
                b = np.matrix(x - mu[k])
                v = np.linalg.solve(L[k], b.transpose())
                y = np.linalg.solve(L[k].transpose(), v)
                t2 = (-1.0/2) * b * y
                delta = t1[k] + t2 + t3[k]
                if delta > lastMax:
                    lastMax = delta
                    h[idx] = k
    else:
        for i in range(d):
            for j in range(d):
                if i != j:
                    sigma[i][j] = 0
        for k in range(C):
            t1[k] = -(1.0/2) * np.log(np.linalg.det(sigma[:,:,k]))
        for idx, x in enumerate(X):
            lastMax = -sys.maxint - 1
            delta = np.zeros(shape=C)
            for k in range(C):
                b = np.matrix(x - mu[k])
                y = np.linalg.solve(sigma[:,:,k], b.transpose())
                t2 = (-1.0/2) * b * y
                delta = t1[k] + t2 + t3[k]
                if delta > lastMax:
                    lastMax = delta
                    h[idx] = k

    return h


# ## Test the Maximum Likelihood estimates
#
# Call `genBlobs` and `plotGaussian` to verify your estimates.

X, labels = genBlobs(centers=5)
mu, sigma = mlParams(X,labels)
# plotGaussian(X,labels,mu,sigma)


# ## Boosting functions to implement
#
# The lab descriptions state what each function should do.

# in:       X - N x d matrix of N data points
#      labels - N x 1 vector of class labels
#           T - number of boosting iterations
# out: priors - length T list of prior as above
#         mus - length T list of mu as above
#      sigmas - length T list of sigma as above
#      alphas - T x 1 vector of vote weights
def trainBoost(X, labels,T=5,covdiag=True):
    def unix(W):
        mu, sigma = mlParams(X, labels, W)
        prior = computePrior(labels, W)
        hti = [False] * N
        h = classify(X, prior, mu, sigma, covdiag)
        e = 0
        for i in range(N):
            if h[i] == labels[i]:
                hti[i] = True
                e += W[i] 
        alpha = (1.0/2) * (np.log(1 - e) - np.log(e))
        return mu, sigma, prior, e, alpha, hti

    N = len(X)
    d = len(X[0])
    C = len(set(labels))

    W = np.zeros(shape=(T, N))
    W[0] += 1.0/N

    e = np.zeros(shape=(T))

    mu = np.zeros(shape=(T, C, d))
    sigma = np.zeros(shape=(T, d, d, C))
    prior = np.zeros(shape=(T, C))

    alpha = np.zeros(shape=T)

    for t in range(1, T):
        mu[t-1], sigma[t-1], prior[t-1], e[t-1], alpha[t-1], hti = unix(W[t-1])
        W[t] = W[t-1]
        for i in range(N):
            if hti:
                W[t][i] *= e[t-1]**(-alpha[t-1])
            else:
                W[t][i] *= e[t-1]**alpha[t-1]
        W[t] /= sum(W[t]) # NORMALIZE RECTIFY JUSTIFY

    mu[T-1], sigma[T-1], prior[T-1], e[T-1], alpha[T-1], hti = unix(W[T-1])
    return prior,mu,sigma,alpha

# in:       X - N x d matrix of N data points
#      priors - length T list of prior as above
#         mus - length T list of mu as above
#      sigmas - length T list of sigma as above
#      alphas - T x 1 vector of vote weights
# out:  yPred - N x 1 class predictions for test points
def classifyBoost(X,priors,mus,sigmas,alphas,covdiag=True):
    T = len(priors)
    C = len(priors[0])
    N = len(X)
    h = np.zeros(shape=(T, N))
    for t in range(T):
        h[t] = classify(X, priors[t], mus[t], sigmas[t])

    c = np.zeros(shape=N)
    for idx, x in enumerate(X):
        bestClass = 0
        lastMax = -sys.maxint - 1
        for k in range(C):
            tot = 0
            for t in range(T):
                if h[t][idx] == k:
                    tot += alphas[t]
            if tot > lastMax:
                lastMax = tot
                bestClass = k
        c[idx] = bestClass
        
    return c


# ## Define our testing function
#
# The function below, `testClassifier`, will be used to try out the different datasets. `fetchDataset` can be provided with any of the dataset arguments `wine`, `iris`, `olivetti` and `vowel`. Observe that we split the data into a **training** and a **testing** set.

np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=25)
np.set_printoptions(linewidth=200)

def testClassifier(dataset='iris',dim=0,split=0.7,doboost=False,boostiter=5,covdiag=True,ntrials=100):

    X,y,pcadim = fetchDataset(dataset)

    means = np.zeros(ntrials,);

    for trial in range(ntrials):

        # xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplit(X,y,split)
        xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split)

        # Do PCA replace default value if user provides it
        if dim > 0:
            pcadim = dim
        if pcadim > 0:
            pca = decomposition.PCA(n_components=pcadim)
            pca.fit(xTr)
            xTr = pca.transform(xTr)
            xTe = pca.transform(xTe)

        ## Boosting
        if doboost:
            # Compute params
            priors,mus,sigmas,alphas = trainBoost(xTr,yTr,T=boostiter)
            yPr = classifyBoost(xTe,priors,mus,sigmas,alphas,covdiag=covdiag)
        else:
        ## Simple
            # Compute params
            prior = computePrior(yTr)
            mu, sigma = mlParams(xTr,yTr)
            # Predict
            yPr = classify(xTe,prior,mu,sigma,covdiag=covdiag)

        # Compute classification error
        print "Trial:",trial,"Accuracy",100*np.mean((yPr==yTe).astype(float))

        means[trial] = 100*np.mean((yPr==yTe).astype(float))

    print "Final mean classification accuracy ", np.mean(means), "with standard deviation", np.std(means)


# ## Plotting the decision boundary
#
# This is some code that you can use for plotting the decision boundary
# boundary in the last part of the lab.

def plotBoundary(dataset='iris',split=0.7,doboost=False,boostiter=5,covdiag=True):

    X,y,pcadim = fetchDataset(dataset)
    xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split)
    pca = decomposition.PCA(n_components=2)
    pca.fit(xTr)
    xTr = pca.transform(xTr)
    xTe = pca.transform(xTe)

    pX = np.vstack((xTr, xTe))
    py = np.hstack((yTr, yTe))

    if doboost:
        ## Boosting
        # Compute params
        priors,mus,sigmas,alphas = trainBoost(xTr,yTr,T=boostiter,covdiag=covdiag)
    else:
        ## Simple
        # Compute params
        prior = computePrior(yTr)
        mu, sigma = mlParams(xTr,yTr)

    xRange = np.arange(np.min(pX[:,0]),np.max(pX[:,0]),np.abs(np.max(pX[:,0])-np.min(pX[:,0]))/100.0)
    yRange = np.arange(np.min(pX[:,1]),np.max(pX[:,1]),np.abs(np.max(pX[:,1])-np.min(pX[:,1]))/100.0)

    grid = np.zeros((yRange.size, xRange.size))

    for (xi, xx) in enumerate(xRange):
        for (yi, yy) in enumerate(yRange):
            if doboost:
                ## Boosting
                grid[yi,xi] = classifyBoost(np.matrix([[xx, yy]]),priors,mus,sigmas,alphas,covdiag=covdiag)
            else:
                ## Simple
                grid[yi,xi] = classify(np.matrix([[xx, yy]]),prior,mu,sigma,covdiag=covdiag)

    classes = range(np.min(y), np.max(y)+1)
    ys = [i+xx+(i*xx)**2 for i in range(len(classes))]
    colormap = cm.rainbow(np.linspace(0, 1, len(ys)))

    plt.hold(True)
    conv = ColorConverter()
    for (color, c) in zip(colormap, classes):
        try:
            CS = plt.contour(xRange,yRange,(grid==c).astype(float),15,linewidths=0.25,colors=conv.to_rgba_array(color))
        except ValueError:
            pass   
        xc = pX[py == c, :]
        plt.scatter(xc[:,0],xc[:,1],marker='o',c=color,s=40,alpha=0.5)

    plt.xlim(np.min(pX[:,0]),np.max(pX[:,0]))
    plt.ylim(np.min(pX[:,1]),np.max(pX[:,1]))
    plt.show()


# ## Run some experiments
#
# Call the `testClassifier` and `plotBoundary` functions for this part.

# Example usage of the functions
# prior, mu, sigma, alpha = trainBoost(X, labels)
# print mu
# quit()
ds = 'iris'
cd = False
boots = True
testClassifier(dataset=ds,split=0.7,doboost=boots,boostiter=5,covdiag=cd)
plotBoundary(dataset=ds,split=0.7,doboost=boots,boostiter=5,covdiag=cd)
