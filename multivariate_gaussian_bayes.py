"""
Multivariate Gaussian Bayes
"""

import numpy as np


def make_data():
    k = 3  # 3 classes
    m = 1000
    m1,m2,m3 = 200,300,500
    n = 4 # 4-dimensional data
    x1 = np.random.normal(loc=[0,1,2,3], scale=1, size=(m1,n))
    x2 = np.random.normal(loc=(3,2,1,0), scale=(1,1,1,1), size=(m2,n))
    x3 = np.random.normal(loc=(0,3,2,1), scale=(1,2,3,4), size=(m3,n))
    X = np.concatenate([x1,x2,x3], axis=0)
    y = (0,)*m1 + (1,)*m2 + (2,)*m3
    return(X,y)
    

class MultivariateGaussianBayes:
    def fit(self, X,y):
        yy = tuple(y)
        m = len(yy)
        self.classes = classes = sorted(set(yy))
        
        #priors
        self.priors = {c:yy.count(c)/m for c in classes}
        
        #conditionals = pdf products
        from statistics import pstdev as std, mean
        self.mu = {(j,c):mean(x for x,y in zip(xx,yy) if y==c) for j,xx in enumerate(X.T) for c in classes}
        self.sd = {(j,c):std(x for x,y in zip(xx,yy) if y==c) for j,xx in enumerate(X.T) for c in classes}
        return self
    
    
    @staticmethod
    def pdf(x, mu, sigma):
        from math import exp, pi, sqrt
        return(1 / sqrt(2*pi*sigma**2) * exp(-(x-mu)**2/(2*sigma**2)) )

    def p(self, *args:'prior or conditional as x,j,y'):
        if len(args)==1: 
            y = args[0]
            return self.priors[y]
        elif len(args)==3:
            x,j,y = args
            mean = self.mu[(j,y)]
            sigma = self.sd[(j,y)]
            return self.pdf(x, mean, sigma)
        else: raise ValueError("bad input")
        
    def posterior(self, y,x):
        from functools import reduce
        from operator import mul
        classes, p = self.classes, self.p
        numerator = p(y) * reduce(mul, (p(x,j,y) for j,x in enumerate(x)))
        normalizer = sum(p(c) * reduce(mul, (p(x,j,c) for j,x in enumerate(x))) for c in classes)
        return numerator/normalizer
 
    def predict_probabilities(self, X):
        return [[self.posterior(y,xx) for y in self.classes] for xx in X]
                   
    def predict(self, X):
        ppred = self.predict_probabilities(X)
        ypred = [pp.index(max(pp)) for pp in ppred]
        return ypred
    
    def get_accuracy(self, X,y):
        ytrue, ypred = y, self.predict(X)
        return sum((ytrue==ypred for ytrue,ypred in zip(ytrue,ypred)))/len(ytrue)


#======================================================================================
    
X,y = make_data()
    
md = MultivariateGaussianBayes().fit(X,y)
ypred = md.predict(X)
ppred = md.predict_probabilities(X)
accuracy = md.get_accuracy(X,y)
print(accuracy)


#sklearn
from sklearn.naive_bayes import GaussianNB
md = GaussianNB().fit(X,y)
accuracy = md.score(X,y)
print(accuracy)

P1 = md.predict_proba(X)
P2 = np.array(ppred)
b = np.allclose(P1,P2)
