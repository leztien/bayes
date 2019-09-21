

"""
Multivariate Naive Bayes and Multinomial Naive Bayes (for discreet data)
"""

import numpy as np


def make_data(m=100, n=3, n_classes=2, t:'number of trials'=10):
    mm = np.random.random(n_classes)
    mm = [int(m*p) for p in mm/sum(mm)]
    
    mm_sum = sum(mm)
    if mm_sum<m:
        ix = mm.index(min(mm))
        mm[ix] += (m-mm_sum)
    elif mm_sum > m:
        ix = mm.index(max(mm))
        mm[ix] -= (mm_sum-m)
        
    pp = [np.random.random(n+1) for _ in mm]
    pp = [pp/sum(pp) for pp in pp]
    
    XX = [np.random.multinomial(n=t, pvals=p, size=m) for m,p in zip(mm,pp)]
    X = np.concatenate(XX, axis=0)[:,:-1]
    y = sum([[c]*m for m,c in zip(mm, range(n_classes))], [])
    return X,y

#============================================================================



from functools import reduce
from operator import mul
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
    def fit(self, X,y):
        super().fit(X,y)
        yy,cc,vv,m,n,t,priors = self.yy,self.cc,self.vv,self.m,self.n,self.t,self.priors
        conditionals = {(v,j,c): len([x for x,y in zip(X[:,j],yy) if y==c and x==v])/yy.count(c)   for j in range(n) for v in vv for c in cc}
        
        def posterior(y,x):
            numerator = priors[y] * reduce(mul, (conditionals[(x,j,y)] for j,x in enumerate(x)))
            normalizer = sum( priors[c]*reduce(mul, (conditionals[(x,j,c)] for j,x in enumerate(x))) for c in cc  ) 
            return numerator / normalizer
        self.posterior = posterior
        return self
    


class PseudoMultinomialNaiveBayes(Abstract):
    def fit(self, X,y):
        from statistics import mean
        from scipy.stats import binom
        
        super().fit(X,y)
        yy,cc,vv,m,n,t,priors = self.yy,self.cc,self.vv,self.m,self.n,self.t,self.priors

        pmf = binom.pmf
        
        trial_probs = {(j,c):mean([x for x,y in zip(X[:,j].tolist() ,yy) if y==c]) / max([x for x,y in zip(X[:,j].tolist() ,yy) if y==c]) 
                            for j in range(n) for c in cc}
        pmf_conditionals = {(x,j,y): pmf(x, t, trial_probs[(j,y)])  
                        for j in range(n) for x in vv for y in cc}
        
        def posterior(y,x):
            numerator = priors[y] * reduce(mul, (pmf_conditionals[(x,j,y)] for j,x in enumerate(x)))
            normalizer = sum( priors[c]*reduce(mul, (pmf_conditionals[(x,j,c)] for j,x in enumerate(x))) for c in cc  ) 
            return numerator / normalizer
        self.posterior = posterior
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


#======================================================================

X,y = make_data(m=100, n=4, n_classes=3, t=10)

  
md = MultivariateNaiveBayes()
md.fit(X,y)
P = md.probabilities(X)
ypred = md.predict(X)
accuracy = md.accuracy(X,y)
print("synthetic dataset:")
print(accuracy.round(2), "({} accuracy)".format(md.__class__.__name__))


md = MultinomialNaiveBayes()
md.fit(X,y)
P = md.probabilities(X)
ypred = md.predict(X)
accuracy = md.accuracy(X,y)
print(accuracy.round(2), "({} accuracy)".format(md.__class__.__name__))


from sklearn.naive_bayes import MultinomialNB
md = MultinomialNB(alpha=1E-9)
md.fit(X,y)
ypred = md.predict(X)
accuracy = md.score(X,y)
print(accuracy.round(2), "({} accuracy)".format(md.__class__.__name__))


#DIGITS
print("\nthe digits dataset:")
from sklearn.datasets import load_digits
X,y = load_digits(n_class=3, return_X_y=True)
X = X.astype('uint16')


md = MultivariateNaiveBayes().fit(X,y)
accuracy = md.accuracy(X,y)
print(accuracy.round(2), "({} accuracy)".format(md.__class__.__name__))


md = MultinomialNaiveBayes().fit(X,y)
accuracy = md.accuracy(X,y)
print(accuracy.round(2), "({} accuracy)".format(md.__class__.__name__))


md = MultinomialNB(alpha=1E-9).fit(X,y)
accuracy = md.score(X,y)
print(accuracy.round(2), "({} accuracy)".format(md.__class__.__name__))

