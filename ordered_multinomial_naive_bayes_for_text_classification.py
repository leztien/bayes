
"""
(ordered) multinomial naive bayes for text classification
"""

from functools import reduce
from operator import mul, add
import abc
class Abstract(abc.ABC):
    @abc.abstractmethod
    def fit(self, X,y):
        yy = tuple(y)
        (m,n) = X.shape
        cc = sorted(set(yy))
        t = int(X.sum(axis=1).max())
        vv = tuple(range(t+1))
        priors = {c:yy.count(c)/m for c in cc}
        [self.__dict__.__setitem__(k,v) for k,v in locals().items() if k!='self']  # ...with the help of this line
        return self
        
    def probabilities(self, X):
        P = [[self.posterior(y,x) for y in self.cc] for x in X]
        return np.array(P)
    
    def predict(self, X):
        return [pp.argmax() for pp in self.probabilities(X)]
    
    def accuracy(self, X,y):
        ypred = self.predict(X)
        return sum(y1==y2 for y1,y2 in zip(y,ypred))/len(y)
    

class MultivariateNaiveBayes(Abstract):
    def __init__(self, avoid_numerical_underflow=True):
        self.avoid_numerical_underflow = avoid_numerical_underflow
    
    def fit(self, X,y):
        super().fit(X,y)
        yy,cc,vv,m,n,t,priors = self.yy,self.cc,self.vv,self.m,self.n,self.t,self.priors
        conditionals = {(v,j,c): len([x for x,y in zip(X[:,j],yy) if y==c and x==v])/yy.count(c)   for j in range(n) for v in vv for c in cc}
        
        def posterior(y,x):
            numerator = priors[y] * reduce(mul, (conditionals[(x,j,y)] for j,x in enumerate(x)))
            normalizer = sum( priors[c] * reduce(mul, (conditionals[(x,j,c)] for j,x in enumerate(x))) for c in cc  ) 
            return numerator / normalizer
        
        def posterior_log(y,x):
            from math import log
            numerator = log(priors[y]) + reduce(add, ( log(conditionals[(x,j,y)] or 0.0001) for j,x in enumerate(x)))
            normalizer = -sum( log(priors[c]) + reduce(add, ( log(conditionals[(x,j,c)] or 0.0001) for j,x in enumerate(x))) for c in cc  )             
            return (numerator / normalizer + 1) / 2
        
        self.posterior = posterior_log if self.avoid_numerical_underflow else posterior
        return self
    



class MultinomialNaiveBayes(Abstract):
    def fit(self, X,y):
        super().fit(X,y)
        yy,cc,vv,m,n,t,priors = self.yy,self.cc,self.vv,self.m,self.n,self.t,self.priors
        
        from scipy.stats import multinomial
        mpmf = multinomial.pmf
        
        ytrue = np.array(yy)
        d = {c:X[ytrue==c,:].mean(0)/t for c in cc}
        mpmf_trial_probabilities = {c:[*v, 1-sum(v)] for c,v in d.items()}
        assert all(sum(mpmf_trial_probabilities[i])==1 for i in range(2))
        
        def posterior(y,x):
            xx = [*x, t-sum(x)]
            numerator = priors[y] * mpmf(xx, t, mpmf_trial_probabilities[y])
            normalizer = sum(priors[c] * mpmf(xx,t, mpmf_trial_probabilities[c]) for c in cc)
            return numerator / normalizer
        self.posterior = posterior
        return self
