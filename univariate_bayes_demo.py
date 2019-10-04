"""
demo on three prediction techniques for univariate bayes:
1. from contingency table
2. mechanical univariate bayes
3. binomial univariate bayes
"""

def make_contingency_table(x,y):
    from pandas import Series, crosstab
    sr = Series(data=y, index=x)
    df = crosstab(index=sr.index, columns=sr.values)
    df.index.name = None; df.columns.name = None
    return df


def contingency_table_to_raw_observations(table):
    from itertools import product
    from numpy import array
    
    try: table = table.values
    except AttributeError: pass
    
    try: table = table.tolist()
    except AttributeError: pass
    
    pairs = list(product(range(len(table)), range(len(table[0]))))
    counts = sum(table, [])
    mx = [[pair]*count for pair,count in  zip(pairs, counts)]
    mx = sum(mx, [])
    X = array(mx, dtype='uint8')
    return X


def predict_from_contingency_table(X, contingecy_table):
    ypred = df.values.argmax(axis=1).take([sorted(set(X)).index(x) for x in X])
    return ypred


def predict_probabilities(x,y):   #mechanical likelihoods
    from numpy import array
    xx = tuple(x)
    yy = tuple(y)
    cc = sorted(set(yy))
    vv = sorted(set(xx))
    
    priors = {c:yy.count(c)/len(yy) for c in cc}
    p = {(v,c):len([x for x,y in zip(xx,yy) if y==c and x==v])/yy.count(c) for v in vv for c in cc}
    p.update(priors)
    
    posterior = lambda y,x : p[y]*p[(x,y)] / sum(p[c]*p[(x,c)] for c in cc)
    P = [[posterior(c,x) for c in cc] for x in xx]
    return array(P, dtype='float64')


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


def predict_probabilities_binomial(x,y): #univariate bayes (binomial)
    from numpy import array
    
    yy = tuple(y)
    cc = sorted(set(yy))
    xx = [sorted(set(x)).index(xi) for xi in x]
    
    μ = {c:sum(x for x,y in zip(xx,yy) if y==c)/yy.count(c) for c in cc}
    t = max(xx)
    p = {c:μ[c]/t for c in cc}
    
    from math import factorial     #scipy.stats.binom.pmf
    pmf = lambda x,t,p : (factorial(t)//(factorial(t-x)*factorial(x))) * p**x * (1-p)**(t-x)
    priors = lambda y : yy.count(y)/len(yy)
    
    
    posterior = lambda y,x : priors(y)*pmf(x,t,p[y]) / sum(priors(c)*pmf(x,t,p[c]) for c in cc)
    P = [[posterior(c,x) for c in cc] for x in xx]
    return array(P)


#=============================================================================

#data:
a,b,c = 'abc'
x = (a,a,b,c,b,b,c,c,c,c)
y = (0,0,0,0,1,1,1,1,1,1)


#predict from contingency table
print("\nCONTINGENCY TABLE:")
contingency_table = df = make_contingency_table(x,y)

#predict
ypred = predict_from_contingency_table(x, contingecy_table=df)
print(ypred)

#evaluate
accuracy = sum(y1==y2 for y1,y2 in zip(y,ypred))/len(ypred)
print("accuracy from contingency table =", accuracy)


#univariate bayes (mechanical)
print("\nunivariate bayes (mechanical):".upper())
P = predict_probabilities(x,y)
ypred = P.argmax(1)
accuracy = sum(y1==y2 for y1,y2 in zip(y,ypred))/len(ypred)
print(ypred)
print("accuracy of univariate bayes (mechanical) =", accuracy)
J = cost(y, P)
print("cross-entropy per observation (mechanical)=", round(J,3))


#univariate bayes (binomial)
print("\nunivariate bayes (binomial):".upper())
P = predict_probabilities_binomial(x,y)
ypred = P.argmax(1)
accuracy = sum(y1==y2 for y1,y2 in zip(y,ypred))/len(ypred)
print("accuracy of univariate bayes (binomial) =", accuracy)
J = cost(y, P)
print("cross-entropy per observation (binomial)=", round(J,3))



"""TEST WITH SYNTHETIC DATA"""
def make_data():
    from numpy.random import binomial
    from numpy import concatenate
    (m,n) = 100,1
    t = 10
    pp = (p0,p1,p2) = (0.1, 0.5, 0.9)    
    mm = (20, 30, 50)
    
    X = concatenate([binomial(n=t, p=p, size=m) for p,m in zip(pp,mm)])
    y = sum([[c]*m for c,m in zip(range(len(pp)),mm)], [])
    return(X.tolist(), y)
    

x,y = make_data()

#------------
print("\n\nSYNTHETIC DATA:")
df = make_contingency_table(x,y)
ypred = predict_from_contingency_table(x, df)
accuracy = sum(y1==y2 for y1,y2 in zip(y,ypred))/len(ypred)
print("accuracy from contingency table =", accuracy)


P = predict_probabilities(x,y)
ypred = P.argmax(1)
accuracy = sum(y1==y2 for y1,y2 in zip(y,ypred))/len(ypred)
J = cost(y, P)
print("accuracy of univariate bayes (mechanical) =", accuracy, end='\t')
print("cross-entropy per observation =", round(J,3))

P = predict_probabilities_binomial(x,y)
ypred = P.argmax(1)
accuracy = sum(y1==y2 for y1,y2 in zip(y,ypred))/len(ypred)
J = cost(y, P)
print("accuracy of univariate bayes (binomial) =", accuracy, end='\t\t')
print("cross-entropy per observation =", round(J,3))


from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from numpy import array
Xoh = OneHotEncoder(categories='auto').fit_transform(array(x).reshape(-1,1))
md = BernoulliNB(alpha=1E-9).fit(Xoh, y)
print("sklearn Bernoulli Naive Bayes accuracy =", md.score(Xoh, y), end="\t\t")
P = md.predict_proba(Xoh)
J = cost(y, P)
print("cross-entropy per observation =", round(J,3))
