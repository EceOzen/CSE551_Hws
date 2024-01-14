# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 21:24:10 2024

@author: Ece
"""

import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate a random system of twenty equations with three unknowns
A = np.random.rand(20, 3)
b = np.random.rand(20)

# Calculate the pseudo-inverse matrix
A_pseudo_inv = np.linalg.pinv(A)

# Verify properties in Theorem 2 (Minimal Solution Theorem)

print("Random generated matrix:")
print(A)
# Property 1: A * A_pseudo_inv * A should be equal to A
property_1_result = np.allclose(A, A @ A_pseudo_inv @ A)

# Property 2: A_pseudo_inv * A * A_pseudo_inv should be equal to A_pseudo_inv
property_2_result = np.allclose(A_pseudo_inv, A_pseudo_inv @ A @ A_pseudo_inv)

# Property 3: (A * A_pseudo_inv).T should be equal to A * A_pseudo_inv
property_3_result = np.allclose((A @ A_pseudo_inv).T, A @ A_pseudo_inv)

# Print results
print("Property 1:", property_1_result)
print("Property 2:", property_2_result)
print("Property 3:", property_3_result)

# Given matrix
A_7 = np.array([
    [-85, -55, -115],
    [-35, 97, -167],
    [79, 56, 102],
    [63, 57, 69],
    [45, -8, 97.5]
])

# Calculate the pseudo-inverse matrix
A_pseudo_inv = np.linalg.pinv(A_7)

# Verify properties in Theorem 2 (Minimal Solution Theorem)

print("\nMatrix in Section 7 is:")
print(A_7)

# Property 1: A * A_pseudo_inv * A should be equal to A
property_1_result = np.allclose(A_7, A_7 @ A_pseudo_inv @ A_7)

# Property 2: A_pseudo_inv * A * A_pseudo_inv should be equal to A_pseudo_inv
property_2_result = np.allclose(A_pseudo_inv, A_pseudo_inv @ A_7 @ A_pseudo_inv)

# Property 3: (A * A_pseudo_inv).T should be equal to A * A_pseudo_inv
property_3_result = np.allclose((A_7 @ A_pseudo_inv).T, A_7 @ A_pseudo_inv)

# Print results
print("Property 1:", property_1_result)
print("Property 2:", property_2_result)
print("Property 3:", property_3_result)
