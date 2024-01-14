# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 21:16:53 2024

@author: Ece
"""

import numpy as np
from scipy.optimize import lsq_linear

A1 = np.array([[1, 2], [2, 3], [3, 4]])
b1 = np.array([5, 6, 7])

x1 = lsq_linear(A1, b1).x

A2 = np.array([[1, 2], [2, 3], [3, 4]])
b2 = np.array([5, 6, 7])

x2 = lsq_linear(A2, b2, bounds=(0, np.inf)).x

A3 = np.array([[1, 2], [2, 3], [3, 4]])
b3 = np.array([5, 6, 7])

x3 = lsq_linear(A3, b3, bounds=(0, 10)).x

A4 = np.array([[1, 2], [2, 3], [3, 4]])
b4 = np.array([5, 6, 7])

x4 = lsq_linear(A4, b4, method='trf').x

print("A1:")
print(A1)
print("\nb1:")
print(b1)
print("\nSolution1:")
print(x1)
print("\nA2:")
print(A2)
print("\nb2:")
print(b2)
print("\nSolution2:")
print(x2)
print("\nA3:")
print(A3)
print("\nb3:")
print(b3)
print("\nSolution3:")
print(x3)
print("\nA4:")
print(A4)
print("\nb4:")
print(b4)
print("\nSolution4:")
print(x4)
