

"""
Univariate Discreet Bayes Multiclass Classifier
"""


def make_data():
    s = """\
12 1 12
5 3 17
2 20 3
1 6 18
"""

    s = """\
20 0 5
0 25 0
0 5 20
0 0 25
"""
    
    mx = [[int(x) for x in l.split(' ')] for l in s.strip().split('\n')]
    n_rows = len(mx)
    n_cols = len(mx[0]) 
    
    y = (0,1,2)
    x = "abcd"
    
    raw_table = [((x[row],y[col]),)*mx[row][col] for row in range(n_rows) for col in range(n_cols)]
    raw_table = sum(raw_table, ())
    
    from operator import itemgetter
    xx,yy = [[itemgetter(col)(t) for t in raw_table] for col in range(2)]
    return(xx,yy)
    

def encode(xx):
    from numpy import array
    classes = sorted(frozenset(xx))
    xx = [classes.index(x) for x in xx]
    X = array(xx, dtype='uint8').reshape(-1,1)
    return X


def onehotize(xx):
    from numpy import array
    classes = sorted(frozenset(xx))
    m = len(xx)
    n = len(frozenset(xx))
    mx = [[0,]*n for _ in range(m)]
    [mx[row].__setitem__(classes.index(x),1) for row,x in enumerate(xx)]
    X = array(mx, dtype='uint8')
    return(X)
    
#==============================================================================
    
xx,yy = make_data()    

import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
X = encode(xx)
y = yy
md = MultinomialNB(alpha=1E-9).fit(X,y)
accuracy = md.score(X,y)
print(accuracy)


X = onehotize(xx)
md = BernoulliNB(alpha=1E-9).fit(X,y)
accuracy = md.score(X,y)
print(accuracy)



#my model
xx = tuple(xx)
yy = tuple(yy)
m = len(yy)

classes = sorted(set(y))
p = {(x,c):len([None for v,y in zip(xx,yy) if y==c and v==x])/yy.count(c) for x in xx for c in classes}
p.update({c:yy.count(c)/m for c in classes})
p.update({x:xx.count(x)/m for x in sorted(set(xx))})

func = lambda y,x : p[y]*p[(x,y)] / p[x]
ppred = [[func(y,x) for y in classes] for x in xx]
ypred = [pp.index(max(pp)) for pp in ppred]
accuracy = sum(ytrue==ypred for ytrue,ypred in zip(yy,ypred)) / len(yy)
print(accuracy)
