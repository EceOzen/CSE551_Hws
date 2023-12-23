# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:16:06 2023

@author: Ece
"""

import numpy as np

def Jacobi(A, b, x0=None, tol=1e-6, max_iter=1000):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    x_new = np.zeros(n)

    for k in range(max_iter):
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x_new - x) < tol:
            return x_new

        x = x_new.copy()

    raise ValueError("Jacobi method did not converge within the specified number of iterations.")

def Gauss_Seidel(A, b, x0=None, tol=1e-6, max_iter=1000):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()

    for k in range(max_iter):
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x[i] = (b[i] - sigma) / A[i, i]

        if np.linalg.norm(np.dot(A, x) - b) < tol:
            return x

    raise ValueError("Gauss-Seidel method did not converge within the specified number of iterations.")

def SOR(A, b, omega, x0=None, tol=1e-6, max_iter=1000):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()

    for k in range(max_iter):
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i, i]

        if np.linalg.norm(np.dot(A, x) - b) < tol:
            return x

    raise ValueError("SOR method did not converge within the specified number of iterations.")

# Test program for solving CP 8.2.2
def test_linear_system_methods():
    A = np.array([[7.0, 1.0, -1.0, 2.0],
                  [1.0, 8.0, 0.0, -2.0],
                  [-1.0, 0.0, 4.0, -1.0],
                  [2.0, -2.0, -1.0, 6.0]])
    
    b = np.array([3.0, -5.0, 4.0, -3.0])

    print("Original matrix A:")
    print(A)

    print("\nOriginal vector b:")
    print(b)

    # Solve using Jacobi method
    x_jacobi = Jacobi(A, b)
    print("\nSolution using Jacobi method:")
    print(x_jacobi)

    # Solve using Gauss-Seidel method
    x_gs = Gauss_Seidel(A, b)
    print("\nSolution using Gauss-Seidel method:")
    print(x_gs)

    # Solve using SOR method
    omega = 1.1  # Adjust omega as needed
    x_sor = SOR(A, b, omega)
    print("\nSolution using SOR method:")
    print(x_sor)

if __name__ == "__main__":
    test_linear_system_methods()
