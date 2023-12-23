# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:07:32 2023

@author: Ece
"""

import numpy as np

def Naive_Gauss(A, b):
    n = len(b)
    for k in range(n-1):
        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] -= factor * A[k, j]
            b[i] -= factor * b[k]

def Gauss(A, b):
    n = len(b)
    scale = np.max(np.abs(A), axis=1)
    for k in range(n-1):
        pivot = np.argmax(np.abs(A[k:n, k]) / scale[k:n]) + k
        A[[k, pivot]] = A[[pivot, k]]
        b[k], b[pivot] = b[pivot], b[k]

        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] -= factor * A[k, j]
            b[i] -= factor * b[k]

def Solve(A, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

# Test program for CP 7.2.1
def test_linear_system():
    A = np.array([[0.4096, 0.1234, 0.3678, 0.2943],
                  [0.2246, 0.3872, 0.4015, 0.1129],
                  [0.3645, 0.1920, 0.3781, 0.0643],
                  [0.1784, 0.4002, 0.2786, 0.3927]])
    

    b = np.array([0.4043, 0.1550, 0.4240, 0.2557])

    print("Original matrix A:")
    print(A)

    print("\nOriginal vector b:")
    print(b)

    # Solve using Naive Gauss elimination
    A1 = A.copy()
    b1 = b.copy()
    Naive_Gauss(A1, b1)
    x1 = Solve(A1, b1)

    print("\nSolution using Naive Gauss elimination:")
    print(x1)

    # Solve using Gaussian elimination with scaled partial pivoting
    A2 = A.copy()
    b2 = b.copy()
    Gauss(A2, b2)
    x2 = Solve(A2, b2)

    print("\nSolution using Gaussian elimination with scaled partial pivoting:")
    print(x2)

if __name__ == "__main__":
    test_linear_system()
