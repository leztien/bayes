
"""
default dictionary of (prior, conditional and marginal) probabilities for univariate bayes analysis
the probabilities are added to the dictionary on the go i.e. on demand
"""

import numpy as np
def make_data():
    """binary univariate data of 10 values from two binomial distributions"""
    m0,m1 = 400,600   # m = 1000
    n = 1
    t = 10
    
    x0 = np.random.binomial(n=t, p=0.5, size=m0)
    x1 = np.random.binomial(n=t, p=0.1, size=m1)
    xx = np.concatenate([x0,x1])
    
    #enlabel the x-values
    from string import ascii_lowercase
    g = zip(range(max(xx)+1), ascii_lowercase[:max(xx)+1])
    d = {k:v for k,v in g}
    xx = [d[x] for x in xx]
    
    yy = [0]*m0 + [1]*m1
    return xx,yy


from collections import defaultdict
class BayesProbabilitiesDictionary(defaultdict):   
    """dictionary of probabilities on demand for univariate descrete bayes analysis"""
    def __init__(self, x:'array of x values', y:'array of target values'):
        self.default_factory = self._func
        self.xx = tuple(x)
        self.yy = tuple(y)
        self.cc = sorted(set(self.yy))
        #calculate the priors:
        m = len(yy)
        d = {c:yy.count(c)/m for c in sorted(set(yy))}
        self.update(d)
    
    def _func(self, k:'as tuple'):  # calculates conditionals and marginals
        if isinstance(k, tuple) and len(k)==2:  # calculates conditionals
            v,c = k
            g = zip(self.xx, self.yy)
            p = len([x for x,y in g if x==v and y==c]) / self.yy.count(c)
        else:  # calculates marginals
            p = sum(self[c]*self[(k,c)] for c in self.cc)
        self.__setitem__(k,p)
        return(p)

    
    def __missing__(self, k):
        v = self.default_factory.__call__(k)
        self.__setitem__(k,v)
        return self.__getitem__(k)
    
    def _numerator(self, x,y):
        return self[y] * self[x,y]
    
    def predict(self, x):
        scores = [self._numerator(x,c) for c in self.cc]
        ypred = scores.index(max(scores))
        return ypred
    
    def posterior(self, y,x):
        numerator = self._numerator(x,y)
        normalizer = self[x]
        return numerator/normalizer
        

#================================================================================


X,y = make_data()
    
d = BayesProbabilitiesDictionary(X,y)

p = d[1]        #prior
p = d[('a',1)]  #conditional
p = d['b']      #marginal

ypred = [d.predict(x) for x in X]
accuracy = sum(y1==y2 for y1,y2 in zip(y,ypred))/len(y)
print("my accuracy", accuracy)

P = [[d.posterior(y,x) for y in sorted(set(y))] for x in X]
P = np.array(P)

from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import OneHotEncoder
X = np.array(X).reshape(-1,1)
X = OneHotEncoder(categories='auto').fit_transform(X)
md = BernoulliNB(alpha=1E-9).fit(X,y)
ypred = md.predict(X)
accuracy = md.score(X,y)
print("sklearn accuracy", accuracy)


