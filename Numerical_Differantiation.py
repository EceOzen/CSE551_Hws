# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:20:53 2023

@author: Ece
"""

import numpy as np

def Derivative(f, x, n=5):
    """
    Compute the nth derivative of a function at a point using Taylor series.

    Parameters:
    - f: Function to differentiate.
    - x: Point at which to compute the derivative.
    - n: Order of the derivative.

    Returns:
    - df_dx: nth derivative of f at x.
    """
    h = 1e-5
    coefficients = [f(x + i * h) / np.math.factorial(i) for i in range(n + 1)]
    taylor_series = np.poly1d(coefficients[::-1])  # Reverse the coefficients

    df_dx = taylor_series.deriv(n)(x)
    return df_dx

def Second_Derivative(f, x, n=5):
    """
    Compute the nth derivative of a function at a point using Taylor series.

    Parameters:
    - f: Function to differentiate.
    - x: Point at which to compute the derivative.
    - n: Order of the derivative.

    Returns:
    - d2f_dx2: nth derivative of f at x.
    """
    h = 1e-5
    coefficients = [f(x + i * h) / np.math.factorial(i) for i in range(n + 1)]
    taylor_series = np.poly1d(coefficients[::-1])  # Reverse the coefficients

    d2f_dx2 = taylor_series.deriv(n).deriv(1)(x)
    return d2f_dx2

def Test_Derivative(f):
    """
    Test the Derivative procedure by solving CP 4.3.1.
    """
    x_values = np.linspace(0, 2*np.pi, 5)
    matrix_of_derivatives = np.zeros((len(x_values), len(x_values)))

    for i, x in enumerate(x_values):
        for j, y in enumerate(x_values):
            matrix_of_derivatives[i, j] = Derivative(f, y)

    print("Matrix of Derivatives:")
    print(matrix_of_derivatives)

def Test_Second_Derivative():
    """
    Test the Second_Derivative procedure by solving CP 4.3.2.
    """
    def f(x):
        return np.sin(x)

    x_point = 0.5
    second_derivative = Second_Derivative(f, x_point)

    print(f"Second Derivative at x={x_point}: {second_derivative}")
    
def test_function_a(x):
    return np.cos(x)

def test_function_b(x):
    return np.arctan(x)

def test_function_c(x):
    return np.abs(x)

if __name__ == "__main__":
    Test_Derivative(test_function_a)
    Test_Derivative(test_function_b)
    Test_Derivative(test_function_c)
    Test_Second_Derivative()
