# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 21:19:54 2024

@author: Ece
"""

import numpy as np
import matplotlib.pyplot as plt

def chebyshev_fit(x, y, degree):
    # Normalize x values to the range [-1, 1]
    x_normalized = 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1
    
    # Compute Chebyshev polynomial basis functions
    basis_functions = [np.cos(k * np.arccos(x_normalized)) for k in range(degree + 1)]
    
    # Form the design matrix
    A = np.column_stack(basis_functions)
    
    # Perform least squares fit
    coefficients, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    
    return coefficients

def chebyshev_polynomial(x, coefficients):
    # Evaluate the Chebyshev polynomial using the coefficients
    result = 0
    for i, coeff in enumerate(coefficients):
        result += coeff * np.cos(i * np.arccos(x))
    return result

# Generate sample data
np.random.seed(42)
x_data = np.linspace(-1, 1, 50)
y_data = 2 * x_data**2 - 1 + 0.2 * np.random.randn(len(x_data))

# Choose the degree of the Chebyshev polynomial
degree = 4

# Perform Chebyshev polynomial fit
coefficients = chebyshev_fit(x_data, y_data, degree)

# Generate fitted curve
x_fit = np.linspace(-1, 1, 100)
y_fit = chebyshev_polynomial(x_fit, coefficients)

# Plot the original data and the fitted curve
plt.scatter(x_data, y_data, label='Original Data')
plt.plot(x_fit, y_fit, label=f'Chebyshev Polynomial Fit (Degree {degree})', color='red')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Least Squares Chebyshev Polynomial Fit')
plt.show()
