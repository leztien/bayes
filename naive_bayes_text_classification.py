# the data created is not good for naive bayes
# but the NB algorith works ok


"""
Naive Bayes for text classification (with a Bag-Of-Words (?))
"""

import re
from collections import Counter
from itertools import groupby
from random import random, choice, shuffle
from collections import UserString
from math import log as ln


PATH = r"/home/linux-ubuntu/Datasets/Bible.txt"


class MarkovModel:
    def tokenize(path):
        with open(PATH, mode='r', encoding='utf-8') as f:
            text = f.read()
        pattern = re.compile(r"[0-9\*\[\]\(\)\s]+")
        text = pattern.sub(' ', text).replace(chr(160), '')
        return text.split()  #list
    
    def ngramize(seq, n=3):
        assert n>=2, "n must be greater than 1"
        return [tuple(seq[i+j] for j in range(n)) for i in range(0, len(seq)-(n-1))]
    
    def compute_distributions(ngrams):
        counter = Counter(ngrams)
        g = groupby(sorted(counter.items(), key=lambda t: t[0]), key=lambda t: t[0][:-1])
        distributions = dict()
        for key, gen in g:
            values, counts = zip(*sorted(((e[0][-1], e[-1]) for e in gen), key=lambda t: t[0]))
            total = sum(counts)
            probs = tuple(count/total for count in counts)
            distributions[key] = {'values': values, 'probs': probs}
        return distributions
    
    def sample_distribution(distributions, key:'ngram', distort=False):        
        values = distributions[key]['values']
        probs = distributions[key]['probs']
        
        if distort and max(probs) < 1.0:
            pp = [1-p for p in probs]
            total = sum(pp)
            probs = [p/total for p in pp]
        r = random()
        cum = 0.0
        for v,p in zip(values, probs):
            cum += p
            if cum > r:
                return v
        return v
    
    def markov_chain(initial_state, length, distributions, distort=False):
        state = tuple(initial_state)
        l = list(state)
        for _ in range(length):
            word = MarkovModel.sample_distribution(distributions, state, distort)
            l.append(word)
            state = tuple(l[-len(state):])
        return str.join(' ', l)
    
    def make_data(m: "number of documents",
                  n: "number of words in each document"):
        N = 3  # N-gram
        tokens = MarkovModel.tokenize(PATH)
        ngrams = MarkovModel.ngramize(tokens, N)
        distributions = MarkovModel.compute_distributions(ngrams)
        
        
        MarkovModel.vocabulary = tuple(set(tokens))
        
        start = choice(ngrams)[:-1]
        
        snippets = []
        labels = []
        
        for i in range(m):
            label = int(bool(i <= m//2))
            generated_text = MarkovModel.markov_chain(
                initial_state=start, 
                length=n,
                distributions=distributions,
                distort=bool(label)
                )
            snippets.append(generated_text)
            labels.append(label)
        return (snippets, labels)

#_____________________________________________________________________________

def tokenize(document):
    STOP_WORD_LEN = 2
    pattern = re.compile(r"\b\w+\b")
    return tuple(w.lower() for w in pattern.findall(document) if len(w) > STOP_WORD_LEN)


class Probabilities:
    def __init__(self, documents, labels):
        self.probs = {}
        self.vocabulary = self.construct_vocabulary(documents)
        self.alpha = 1.0  # Laplace smoother
        self._compute_priors(labels)
        self._compute_conditionals(documents, labels)
        
    def _compute_priors(self, labels):
        labels = tuple(labels)
        m = len(labels)
        self.classes = sorted(set(labels))
        priors = {c: labels.count(c) / m for c in self.classes}
        assert 0.999 < sum(priors.values()) < 1.001, "must sum to 1"
        self.probs.update(priors)
    
    def _compute_conditionals(self, documents, labels):
        documents = tuple(tokenize(d) for d in documents)
        mega_docs = {c:sum((d for d,y in zip(documents, labels) if y==c), ()) for c in self.classes}
        
        conditionals = {(w,c): (mega_docs[c].count(w) + self.alpha) / (len(mega_docs[c]) + len(self.vocabulary))
                        for w in self.vocabulary
                        for c in self.classes}
        self.probs.update(conditionals)
    
    @staticmethod
    def construct_vocabulary(documents):
        if type(documents[0]) is str:
            documents = [tokenize(d) for d in documents]
        documents = (tuple(d) for d in documents)
        return tuple(sorted(set(sum(documents, ()))))

    def __call__(self, *args, **kwargs):
        key = args[0] if len(args)==1 else kwargs.popitem()[-1]
        return self.probs.get(key, 1.0)  # NA value

    class Word(UserString):
        def __init__(self, word):
            self.data = str(word)
        def __or__(self, c):  # dunder method for bitwise-or  '|'
            return (self.data, c)
        def __hash__(self):
            return hash(self.data)
        def __eq__(self, other):
            return hash(self) == hash(other)
    @classmethod
    def convert_string(cls, word):
        return cls.Word(word)


def predict(document, probabilities:"Naive Bayes model"):
    assert type(probabilities) is Probabilities, "the 'Naive Bayes model' must be the probabilities encapsulated in a 'Probabilitiy' class-obkect"
        
    p = probabilities
    C = p.classes
    d = tuple(p.convert_string(w) for w in tokenize(document))
    predictions = dict()
    
    for c in C:
        predictions[c] = ln(p(c)) + sum(ln(p(w|c)) for w in d)
    return max(predictions, key=lambda k: predictions[k])
    
    
def accuracy(ytrue, ypred):
    assert len(ytrue) == len(ypred), "must be of equal length"
    return sum(y1==y2 for y1,y2 in zip(ytrue, ypred)) / len(ytrue)

##############################################################################



# Make data
documents, labels = MarkovModel.make_data(m=300, n=50)



def split_data(documents, labels, test_set=0.2):
    assert len(documents) == len(labels), "must be the same"
    nx = list(range(len(labels)))
    shuffle(nx)
    split = round(len(nx) * (1 - test_set))
    nx_train = nx[:split]
    nx_test = nx[split:]
    train_set, test_set = (tuple(documents[i] for i in nx) for nx in (nx_train, nx_test))
    train_labels, test_labels = (tuple(labels[i] for i in nx) for nx in (nx_train, nx_test))
    assert len(train_set) + len(test_set) == len(documents)
    assert len(train_labels) + len(test_labels) == len(labels)
    return (train_set, test_set, train_labels, test_labels)


docs_train, docs_test, y_train, y_test = split_data(documents, labels)


p = Probabilities(docs_train, y_train)


# Predict
ytrue = y_test
ypred = []

for d in docs_test:
    c = predict(d, p)
    ypred.append(c)


acc = accuracy(ytrue, ypred)
print(f"test set accuracy: {acc:.0%}")


#-------------------------------------

d1 = "china, china japan"
d2 = "china japan. korea"
d3 = "china usa france usa"
d4 = "france usa 'england': japan"
d5 = "england: taiwan"
docs = (d1,d2,d3,d4,d5)
yy = [0,0,1,1,0]

pp = Probabilities(docs, yy)

yp = []
for d in docs:
    c = predict(d, pp)
    yp.append(c)

print(yy, yp, accuracy(yy,yp))

