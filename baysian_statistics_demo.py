
"""
demo on baysian statistics
updating the distribution given (binomial) evidence
"""

import numpy as np
import matplotlib.pyplot as plt


from math import factorial
def posterior(y,x):
    global priors
    x,t = x[0], sum(x)
    p = y
    yy = globals()['y']
    tCx = factorial(t)//(factorial(t-x)*factorial(x))
    likelihood = tCx * p**x * (1-p)**(t-x)
    prior = priors[y]
    normalizer = sum(priors[c]*(tCx * c**x * (1-c)**(t-x)) for c in yy)
    posterior = prior * likelihood / normalizer
    return posterior


def update_priors(priors, evidence):
    for c in priors.keys():
        priors[c] = posterior(c, evidence)


def barplot(probabilities, evidence=None, ticks=False, total_number_of_plots=None):
    fig = plt.gcf()
    total_number_of_plots = total_number_of_plots or globals()['total_number_of_plots']
    globals()['total_number_of_plots'] = total_number_of_plots
    plot_numebr = fig.get_axes().__len__() + 1
    sp = plt.subplot(total_number_of_plots,1,plot_numebr)
    keys,values = zip(*sorted(probabilities.items(), key=lambda t : t[0]))
    sp.bar(x=keys, height=values, width=1/len(keys)*0.8, edgecolor='k', tick_label=keys)
    sp.set_ylim(0, 0.45)
    if not ticks:
        sp.set_xticklabels([]); sp.set_xticks([])
        fig.subplots_adjust(hspace=0.07)
    if evidence:
        p = evidence[0]/sum(evidence)
        sp.text(0.02, 0.6, "evidence:\n{:.0%}".format(p), transform=sp.transAxes, fontsize='small')
    return fig
       
#======================================================================  

#GIVEN and ASSUMPTIONS
H = y = yy = [p/100 for p in range(5,100,10)]
priors = {c:1.0/len(y) for c in y}
barplot(priors, total_number_of_plots=4)

#FIRTS EVIDENCE
E = x = (3,1)   # 3:1
update_priors(priors=priors, evidence=x)
barplot(priors, evidence=x)

#SECOND EVIDENCE
E = x = (3,2)   # 3:2
update_priors(priors=priors, evidence=x)
barplot(priors, evidence=x)

#THIRD EVIDENCE
E = x = (13,7)   # 3:2
update_priors(priors=priors, evidence=x)
barplot(priors, evidence=x, ticks=True)

