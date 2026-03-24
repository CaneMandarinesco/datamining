# classificatore basato su k-nearest neighbors
# obiettivo: classificare forme geometriche a partire da sequenze di punti
# dataset: 200 campioni dove 0 (= non quadrato) ed 1 (= quadrato)
#  

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
import pandas as pd

from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def euclidean_dist(x1, x2):
    pass

class KNN(object):
    def __init__(self, k=5, distance=None):
        self.k = k
        self._dist = distance
        if self._dist == None: self._dist = euclidean_dist

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.tree = KDTree(X)

    def predict(self, x):
        pass

def knn_fromcsv(filename, dirname="dataset"):
    path = os.path.join(dirname, filename)
    df = pd.read_csv(path, header=None, encoding="utf8", sep=',')
    X = []
    y = []

    for x, i in df.iterrows():
        
        pass

    knn = KNN()
    knn.fit(X,y)
    return knn


knn_fromcsv("01-01-quadrati.csv")