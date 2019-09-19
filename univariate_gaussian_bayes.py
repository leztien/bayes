
"""
Univariate Gaussian Bayes
"""

import numpy as np
import matplotlib.pyplot as plt


def make_data():
    n = 100
    n1,n2,n3 = 200,300,500
    
    x1 = np.random.normal(10, 1, size=n1)
    x2 = np.random.normal(15, 1.5, size=n2)
    x3 = np.random.normal(20, 2, size=n3)
    
    x = np.concatenate([x1,x2,x3])
    X = x.reshape(-1,1)
    y = [0]*n1 + [1]*n2 + [2]*n3
    return(X,y)

#=====================================================
    
X,y = make_data()
L = [list() for _ in range(3)]
[L[c].append(x) for x,c in zip(X.ravel(),y)] 
[plt.hist(x, alpha=.6) for x in L]


class UnivariateGaussianBayes:
    def fit(self, X,y):
        from statistics import pstdev as std
        classes = sorted(frozenset(y))
        yy = tuple(y)
        xx = tuple(X.ravel())
        p = {c:yy.count(c)/len(yy) for c in classes}
        mu = {c:sum(x for x,y in zip(xx,yy) if y==c)/yy.count(c) for c in classes}
        sd = {c:std(x for x,y in zip(xx,yy) if y==c) for c in classes}
        [self.__dict__.update(locals())]
    
    def get_probabilities(self,X):
        from scipy.stats import norm
        pdf = norm.pdf
        classes = self.classes
        xx = self.xx
        p = self.p
        mu = self.mu
        sd = self.sd
        func = lambda x,y : p[y]*pdf(x, mu[y], sd[y]) / sum(p[y]*pdf(x, mu[y], sd[y]) for y in classes)
        self.ppred = tuple([[func(x,c) for c in classes] for x in xx])
        return self.ppred
    
    def predict(self, X):
        self.get_probabilities(X)
        ypred = [pp.index(max(pp)) for pp in self.ppred]
        return ypred
    
    def get_accuracy(self, X,y):
        ytrue = y
        ypred = self.predict(X)
        accuracy = sum(ypred==ytrue for ypred,ytrue in zip(ypred,ytrue))/len(ypred)
        return accuracy
        


md = UnivariateGaussianBayes()
md.fit(X,y)
ppred = md.get_probabilities(X)
ypred = md.predict(X)
accuracy = md.get_accuracy(X, y)
print(accuracy)


#sklearn
from sklearn.naive_bayes import GaussianNB
md = GaussianNB().fit(X,y)
accuracy = md.score(X,y)
print(accuracy)

P1 = md.predict_proba(X)
P2 = np.array(ppred)
b = np.allclose(P1,P2)

