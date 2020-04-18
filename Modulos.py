import numpy as np
import pandas as pd
import pickle
from functools import reduce
from scipy import optimize as op
fla = lambda list_of_arrays: reduce(lambda acc, v: np.array([*acc.flatten(), *v.flatten()]),list_of_arrays)
def pdt(thtF, shapes, X, Y):
    tht = ift(thtF, shapes)
    a = pgtF(tht, X)
    return [a[-1] , Y]
def pgtF(tht, X):
    lma = [np.asarray(X)] 
    for i in range(len(tht)):
        lma.append(sgd(np.matmul(np.hstack((np.ones(len(X)).reshape(len(X), 1),lma[i])),tht[i].T)))
    return lma
def cost(thtF, shapes, X, Y):
    a = pgtF(ift(thtF, shapes),X)
    return -(Y * np.log(a[-1]) + (1 - Y) * np.log(1 - a[-1])).sum() / len(X) 
def ift(thtF, shapes):
    lys = len(shapes) + 1
    szs = [shape[0] * shape[1] for shape in shapes]
    stp = np.zeros(lys, dtype=int)
    for i in range(lys - 1):
        stp[i + 1] = stp[i] + szs[i]
    return [thtF[stp[i]: stp[i + 1]].reshape(*shapes[i]) for i in range(lys - 1)]
def cbn(thtF, shapes, X, Y):
    m, lys = len(X), len(shapes) + 1
    tht = ift(thtF, shapes)
    a = pgtF(tht, X)
    deltas = [*range(lys - 1), a[-1] - Y]
    for i in range(lys - 2, 0, -1):
        deltas[i] =  (deltas[i + 1] @ np.delete(tht[i], 0, 1)) * (a[i] * (1 - a[i]))
    ard = []
    for n in range(lys - 1):
        ard.append((deltas[n + 1].T @ np.hstack((np.ones(len(a[n])).reshape(len(a[n]), 1),a[n]))) / m)
    ard = np.asarray(ard)
    return fla(ard)
def sgd(z):
    a = [(1 / (1 + np.exp(-i))) for i in z]
    return np.asarray(a).reshape(z.shape)