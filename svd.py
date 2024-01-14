# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 21:16:23 2024

@author: Ece
"""

import numpy as np

# Generate a random matrix (replace this with your specific matrix)
np.random.seed(42)
m, n = 3, 4
A = np.random.rand(m, n)

# Perform SVD
U, S, VT = np.linalg.svd(A)

print("Original Matrix A:")
print(A)
print("\nU matrix:")
print(U)
print("\nSingular Values (diagonal matrix):")
print(np.diag(S))
print("\nVT matrix:")
print(VT)
