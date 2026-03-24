import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
import pandas as pd

def euclidean_dist(a0, a1):
    return np.linalg.norm(a0 - a1)

# estrae i k piu piccoli in ordine.
def selection_sort(a, x, k=None, key=None):
    if k == None or k > len(a) or k < 0:
        k = len(a)
    if key == None:
        key = lambda y, _: y

    idxs = []
    # estrai ogni volta i k piu piccoli
    # O(nk)
    for i in range(k):
        m = None
        for j, v in enumerate(a):
            if m == None or key(v,x) < key(a[m], x):
                if j not in idxs:
                    m = j
        idxs.append(m)

    return idxs

def mode(a):
    """ 
    calcola la moda: elemento piu frequente di a
    """
    # per ogni item in a, mi da le occorrenze
    # itms ordinato per cnts
    itms, cnts = np.unique(np.array(a), return_counts=True)
    return itms[np.argmax(cnts)], max(cnts)

class KNN:
    def __init__(self, k=5, distance=None):
        self.k = k
        self._dist = euclidean_dist if distance == None else distance

    def fit(self, X, y):
        self.X = X # mat nxd (n campioni, d features)
        self.y = y # array di dim n

    def predict(self, x):
        k_indices = selection_sort(self.X, x, k=5, key=self._dist)
        # ritorna la etichetta
        return mode(self.y[k_indices])

def old_train_test_split(X,y, train_size=0.7):
    n = X.shape[0]

    # scegli randomicamente n*train_size indici.
    train_idxs = np.random.choice(n, size=int(n*train_size), replace=False)
    # il test e' costituito dagli indici non scelti prima 
    test_idxs = np.setdiff1d(np.arange(n), train_idxs)

    # ritorna le opportune liste
    return X[train_idxs], y[train_idxs], X[test_idxs], y[test_idxs]

def train_test_split(X,y, train_size=0.7):
    tags = np.unique(y)

    train_idxs = []
    for t in tags:
        # devo sapere quanti ce ne sono
        n = np.where(y==t)[0].shape[0]
        # replace=False: senza rimpiazzamento, ossia prendi i valori una e una sola volta
        # vengono scelti randomicamente gli indici i t.c. y[i]=t
        train_idxs.append(np.random.choice(np.where(y==t)[0], size=int(n*train_size), replace=False))
        
    train_idxs = np.concatenate(train_idxs)
    test_idxs = np.setdiff1d(np.arange(X.shape[0]), train_idxs)
    
    return X[train_idxs], y[train_idxs], X[test_idxs], y[test_idxs]

# read csv 
s = os.path.join('dataset', '01-02-iris.csv')
df = pd.read_csv(s, header=None, encoding='utf-8')

# store and split values
X = df.iloc[:, 1:2].values # seleziona tutte tranne l'ultima
y = df.iloc[:, 4].values

X_train, y_train, X_test, y_test = train_test_split(X,y)

# create KNN object
knn = KNN()
knn.fit(X_train, y_train)

acc = []
for  i in range(10):
    preds = np.array([knn.predict(x)[0] for x in X_test]) == y_test
    acc.append(np.sum(preds)/preds.shape[0])

print("acc: ", np.sum(acc)/len(acc))