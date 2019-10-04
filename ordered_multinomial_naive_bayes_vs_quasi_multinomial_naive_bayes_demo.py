
"""
demo on ordered multinomial naive bayes vs quasi-multinomial naive bayes
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
    
    for i,p in enumerate(pp):
        l = list(pp[i])
        l.append(l.pop(l.index(max(l))))
        pp[i] = l
        
    pp = [pp/sum(pp) for pp in pp]
    
    XX = [np.random.multinomial(n=t, pvals=p, size=m) for m,p in zip(mm,pp)]
    X = np.concatenate(XX, axis=0)[:,:-1]
    y = sum([[c]*m for m,c in zip(mm, range(n_classes))], [])
    return X,y


def cost(y, probabilities):   #cost via cross-entropy
    from math import log as ln
    try: P = probabilities.tolist()
    except AttributeError: P = probabilities
    cc = sorted(set(y))
    yy = tuple(y)
    Y = [[0]*len(cc) for y in yy]
    [Y[i].__setitem__(y,1) for i,y in enumerate(yy)]
    cross_entropy = -sum(ln(p[y.index(1)]) for y,p in zip(Y,P))/len(Y)
    return(cross_entropy)
    
    
#=================================================================================

X,y = make_data(m=1000, n=5, n_classes=4, t=30)





from sklearn.naive_bayes import MultinomialNB
md = MultinomialNB(alpha=1E-9).fit(X,y)
print("accuracy =", md.score(X,y), end="\t")

P = md.predict_proba(X)
J = cost(y, P)
print("cross-entropy per observation (sklearn)=", round(J,3))


#################################################################


yy = tuple(y)
cc = sorted(set(yy))
m,n = X.shape


priors = {c:yy.count(c) for c in cc}

μ = {(j,c):X[np.array(yy)==c,j].mean() for j in range(n) for c in cc}
quasi_probabilities = {(j,c):X[np.array(yy)==c,j].sum()/yy.count(c) for j in range(n) for c in cc}
μ3 = {(j,c): (X[np.array(yy)==c,j].sum()+1)/(yy.count(c)+n) for j in range(n) for c in cc}

from functools import reduce
from operator import mul
conditional = lambda x,y : reduce(mul, ( quasi_probabilities[(j,y)]**x for j,x in enumerate(x)))

posterior_score = lambda y,x : priors[y]*conditional(x,y)

predict = lambda x : np.array([posterior_score(c,x) for c in cc]).argmax()

ypred = [predict(x) for x in X]
ytrue = yy
accuracy = sum(y1==y2 for y1,y2 in zip(ytrue,ypred))/len(ypred)
print("accuracy (with quasi-probabilities) =", accuracy)




##############################################################################



yy = tuple(y)
cc = sorted(set(yy))
m,n = X.shape


priors = {c:yy.count(c) for c in cc}

event_probabilities = {c:(X[np.array(yy)==c]/X[np.array(yy)==c].sum(1, keepdims=True)).mean(0) for c in cc}
from functools import reduce
from operator import mul, add
from math import log

pmf = lambda x,p : reduce(mul, (p**x for p,x in zip(p,x)))

pmf_log = lambda x,p : reduce(add, (x*log(p) for p,x in zip(p,x)))
posterior_score = lambda y,x : log(priors[y]) + pmf_log(x, event_probabilities[y])
ypred = [np.array([posterior_score(c,x) for c in cc]).argmax() for x in X]
ytrue = yy
accuracy = sum(y1==y2 for y1,y2 in zip(ytrue,ypred))/len(ypred)
print("accuracy (ordered multinomial log) =", accuracy)




def posteriors(x):
    normalizer = sum(priors[c]*pmf(x, event_probabilities[c]) for c in cc )
    numerators = [priors[c] * pmf(x, event_probabilities[c]) for c in cc]
    probabilities = [n/normalizer for n in numerators]
    return probabilities
    

P = np.array([posteriors(x) for x in X])

ypred = P.argmax(1)
ytrue = yy
accuracy = sum(y1==y2 for y1,y2 in zip(ytrue,ypred))/len(ypred)
print("accuracy (ordered multinomial) =", accuracy, end='\t')
J = cost(y, P)
print("cross-entropy per observation (ordered multinomial)=", round(J,3))

