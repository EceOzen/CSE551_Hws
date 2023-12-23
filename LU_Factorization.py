# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:10:29 2023

@author: Ece
"""

import numpy as np

def LU_factorization(A):
    n = len(A)
    L = np.eye(n)
    U = A.copy()

    for k in range(n-1):
        for i in range(k+1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:n] -= factor * U[k, k:n]

    return L, U

def solve_with_LU(A, b):
    L, U = LU_factorization(A)

    # Solve Ly = b for y using forward substitution
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Solve Ux = y for x using backward substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

# Test program for solving CP 7.2.1 with LU factorization
def test_solve_with_LU():
    A = np.array([[0.4096, 0.1234, 0.3678, 0.2943],
                  [0.2246, 0.3872, 0.4015, 0.1129],
                  [0.3645, 0.1920, 0.3781, 0.0643],
                  [0.1784, 0.4002, 0.2786, 0.3927]])
    

    b = np.array([0.4043, 0.1550, 0.4240, 0.2557])

    print("Original matrix A:")
    print(A)

    print("\nOriginal vector b:")
    print(b)

    # Solve using LU factorization
    x = solve_with_LU(A, b)

    print("\nSolution using LU factorization:")
    print(x)

if __name__ == "__main__":
    test_solve_with_LU()
