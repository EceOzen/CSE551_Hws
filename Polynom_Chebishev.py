# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:47:55 2023

@author: Ece
"""

import numpy as np

def Coef(x, y):
    """
    Compute the coefficients of the interpolating polynomial.

    Parameters:
    - x: List of x-coordinates of data points.
    - y: List of corresponding y-coordinates of data points.

    Returns:
    - c: List of coefficients of the interpolating polynomial.
    """
    n = len(x)
    c = [0] * n

    for j in range(n):
        c[j] = y[j]
        for k in range(j - 1, -1, -1):
            c[k] = (c[k + 1] - c[k]) / (x[j] - x[k])

    return c

def Eval(c, x, xp):
    """
    Evaluate the interpolating polynomial at a given point xp.

    Parameters:
    - c: List of coefficients of the interpolating polynomial.
    - x: List of x-coordinates of data points.
    - xp: Point at which to evaluate the polynomial.

    Returns:
    - p: Value of the interpolating polynomial at xp.
    """
    n = len(c)
    p = c[n - 1]
    for k in range(n - 2, -1, -1):
        p = p * (xp - x[k]) + c[k]

    return p

def Test_Equal(f, a, b, n, m):
    """
    Test the accuracy of polynomial interpolation.

    Parameters:
    - f: Function to interpolate.
    - a: Lower bound of the interval.
    - b: Upper bound of the interval.
    - n: Number of evaluation points equally spaced in the interval.
    - m: Number of test points, equally spaced within the interval.

    Prints:
    - Accuracy of the function at the test points.
    """

    # Generate equally spaced evaluation points
    eval_points = np.linspace(a, b, n)
    eval_values = [f(x) for x in eval_points]

    # Generate equally spaced test points
    test_points = np.linspace(a, b, m)
    true_values = [f(x) for x in test_points]

    # Calculate coefficients using Coef
    coefficients = Coef(eval_points, eval_values)

    # Evaluate the polynomial at test points using Eval
    interp_values = [Eval(coefficients, eval_points, x) for x in test_points]

    # Calculate accuracy
    accuracy = np.linalg.norm(np.array(interp_values) - np.array(true_values), np.inf)
    print(f"Accuracy at {m} test points: {accuracy}")
    
def Test_Chebishev(f, a, b, n, m):
    """
    Test the accuracy of polynomial interpolation using Chebyshev points.

    Parameters:
    - f: Function to interpolate.
    - a: Lower bound of the interval.
    - b: Upper bound of the interval.
    - n: Number of Chebyshev points in the interval.
    - m: Number of test points, equally spaced within the interval.

    Prints:
    - Accuracy of the function at the test points.
    """

    # Generate Chebyshev points
    cheb_points = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n))

    # Evaluate function at Chebyshev points
    cheb_values = [f(x) for x in cheb_points]

    # Generate equally spaced test points
    test_points = np.linspace(a, b, m)
    true_values = [f(x) for x in test_points]

    # Calculate coefficients using Coef
    coefficients = Coef(cheb_points, cheb_values)

    # Evaluate the polynomial at test points using Eval
    interp_values = [Eval(coefficients, cheb_points, x) for x in test_points]

    # Calculate accuracy
    accuracy = np.linalg.norm(np.array(interp_values) - np.array(true_values), np.inf)
    print(f"Accuracy at {m} test points using Chebyshev points: {accuracy}")


# Example usage:
def test_function_equal(x):
    return 1 / (x**2 + 1)

def test_function_chebishev1(x):
    return 5 * np.cos(x * np.pi / 20)

def test_function_chebishev2(x):
    return 5 * np.cos((2*x + 1) * np.pi / 42)

# Test case CP 4.2.1
Test_Equal(test_function_equal, -5, 5, 21, 41)
# Test case CP 4.2.2
Test_Chebishev(test_function_chebishev1, 0, 20, 10, 100)
Test_Chebishev(test_function_chebishev2, 0, 20, 10, 100)
