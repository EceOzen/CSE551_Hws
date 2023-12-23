# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:12:14 2023

@author: Ece
"""

import numpy as np

def inverse(A):
    if np.linalg.det(A) == 0:
        raise ValueError("Matrix is singular, and its inverse does not exist.")
    
    n = len(A)
    identity = np.eye(n)
    A_inv = np.linalg.solve(A, identity)
    return A_inv

# Test program for finding the inverse of a matrix
def test_inverse():
    A = np.array([[-0.0001, 5.096, 5.101, 1.853],
                  [0.0, 3.737, 3.740, 3.392],
                  [0.0, 0.0, 0.006, 5.254],
                  [0.0, 0.0, 0.0, 4.567]])
    
   
    print("Original matrix A:")
    print(A)

    # Find the inverse of A
    A_inv = inverse(A)

    print("\nInverse of matrix A:")
    print(A_inv)

if __name__ == "__main__":
    test_inverse()
