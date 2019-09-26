





import numpy as np

def make_data():
    from scipy.stats import bernoulli
    m,n = 1000,4
    m0,m1,m2 = 200,300,500
    
    p0 = (0.1, 0.9, 0.1, 0.5)
    p1 = (0.99999, 0.1, 0.0001, 0.1)
    p2 = (0.3, 0.1, 0.9, 0.7)
    
    X0 = bernoulli.rvs(p=p0, size=(m0,n))
    X1 = bernoulli.rvs(p=p1, size=(m1,n))
    X2 = bernoulli.rvs(p=p2, size=(m2,n))
    
    X = np.concatenate([X0, X1, X2], axis=0)
    y = [0]*len(X0) + [1]*len(X1) + [2]*len(X2)
    return(X,y)

#===========================================================

X,y = make_data()


# the code starting from here can be made into a class
yy = tuple(y)
y = np.array(yy)
cc = sorted(set(yy))
m = len(yy)
n = X.shape[1]

zero_count_rectifier = 1E-9
p = {(x,j,c) : abs((1-x) - X[y==c,j].mean()) or zero_count_rectifier for j in range(n) for c in cc for x in (0,1)}
p.update({c:yy.count(c)/len(yy) for c in cc})  # priors


def predict(x, avoid_numeric_underflow=True):
    from operator import mul,add
    from functools import reduce
    from math import log
    if avoid_numeric_underflow:
        numerators = [log(p[y]) + reduce(add, (log(p[(x,j,y)]) for j,x in enumerate(x))) for y in cc]
    else:
        numerators = [p[y]*reduce(mul, (p[(x,j,y)] for j,x in enumerate(x))) for y in cc]
    ypred = numerators.index(max(numerators))
    return ypred


ytrue = y
ypred = [predict(x) for x in X]

accuracy = sum(y1==y2 for y1,y2 in zip(ytrue,ypred))/len(ypred)
print("my Bernoulli Naive Bayes accuracy =", accuracy)


#sklearn 
from sklearn.naive_bayes import BernoulliNB
md = BernoulliNB(alpha=1E-9).fit(X,y)
accuracy = md.score(X,y)
print("sklearn BernoulliNB accuracy =", accuracy)    

