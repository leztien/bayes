
"""
demo on naive gaussian bayes  vs  multivariate gaussian bayes
"""

import numpy as np
import matplotlib.pyplot as plt

def make_data(m=2000):
    Σ1 = [[2,1], [1,1]]
    Σ2 = [[2,-1],[-1,1]]
    
    X1 = np.random.multivariate_normal(mean=[0]*len(Σ1), cov=Σ1, size=m//2)
    X2 = np.random.multivariate_normal(mean=[0]*len(Σ2), cov=Σ2, size=m//2)
    y = [0]*len(X1) + [1]*len(X2)
    X = np.concatenate([X1,X2], axis=0)
    return(X,y)

#==========================================================================

X,y = make_data()


#naive gaussian bayes
from sklearn.naive_bayes import GaussianNB
md = GaussianNB().fit(X,y)
accuracy = md.score(X,y)
print("naive gaussian bayes =", accuracy.round(2))


#multivariate gaussian bayes model
mask = np.equal(y, 1)
Σ1 = np.cov(X[~mask].T, ddof=0)
μ1 = np.mean(X[~mask], axis=0)

Σ2 = np.cov(X[mask].T, ddof=0)
μ2 = np.mean(X[mask], axis=0)

from scipy.stats import multivariate_normal

d1 = multivariate_normal(mean=μ1, cov=Σ1)
d2 = multivariate_normal(mean=μ2, cov=Σ2)

priors = {c:y.count(c) for c in sorted(set(y))}  
# priors are not necessary because we intentionally have a perfectly balanced data-set

P = [[d1.pdf(x)*priors[y], d2.pdf(x)*priors[y]] for x,y in zip(X,y)]
ypred = [pp.index(max(pp)) for pp in P]
accuracy = sum(y1==y2 for y1,y2 in zip(y,ypred)) / len(ypred)
print("multivariate gaussian bayes acuracy =", round(accuracy,2))


#vizualize
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.scatter(*X.T, c=y, marker='.', cmap=plt.cm.RdBu)
plt.axis('equal')

plt.subplot(122)
plt.scatter(*X.T, c=ypred, marker='.', cmap=plt.cm.RdBu)
