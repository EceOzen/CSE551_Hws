# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:31:26 2023

@author: Ece
"""

import numpy as np
import sys

# Set a higher recursion limit
sys.setrecursionlimit(10**6)

def lower_sum(f, a, b, n):
    delta_x = (b - a) / n
    x_values = np.linspace(a, b, n + 1)
    lower_sum = np.sum([f(x) for x in x_values[:-1]]) * delta_x
    return lower_sum

def upper_sum(f, a, b, n):
    delta_x = (b - a) / n
    x_values = np.linspace(a, b, n + 1)
    upper_sum = np.sum([f(x) for x in x_values[1:]]) * delta_x
    return upper_sum

def composite_trapezoid(f, a, b, n):
    delta_x = (b - a) / n
    x_values = np.linspace(a, b, n + 1)
    trapezoid_sum = np.sum([(f(x) + f(x + delta_x)) / 2 for x in x_values[:-1]]) * delta_x
    return trapezoid_sum

def romberg(f, a, b, n):
    kmax = 5  # Maximum number of iterations
    r = np.zeros((kmax + 1, kmax + 1))
    h = b - a
    r[0, 0] = 0.5 * h * (f(a) + f(b))

    for i in range(1, kmax + 1):
        h = h / 2
        summation = np.sum([f(a + k * h) for k in range(1, 2**i, 2)])
        r[i, 0] = 0.5 * r[i - 1, 0] + h * summation

        for j in range(1, i + 1):
            r[i, j] = r[i, j - 1] + (r[i, j - 1] - r[i - 1, j - 1]) / (4**j - 1)

    return r[kmax, kmax]

def composite_simpson(f, a, b, n):
    n = n * 2  # Double n for Simpson's rule
    delta_x = (b - a) / n
    x_values = np.linspace(a, b, n + 1)
    simpson_sum = delta_x / 3 * (f(x_values[0]) + 4 * np.sum(f(x_values[1:-1:2])) + 2 * np.sum(f(x_values[2:-2:2])) + f(x_values[-1]))
    return simpson_sum

def adaptive_simpson_error(f, a, b, tol=1e-6):
    def recursive_simpson(a, b, fa, fb, fc, tol):
        c = (a + b) / 2
        h = b - a
        d = (a + c) / 2
        e = (c + b) / 2
        fd = f(d)
        fe = f(e)
        sab = h * (fa + 4 * fc + fb) / 6
        sac = h * (fa + 4 * fd + fc) / 6
        scb = h * (fc + 4 * fe + fb) / 6
        sabc = sac + scb
        if abs(sabc - sab) <= 15 * tol:
            return sabc + (sabc - sab) / 15
        else:
            return recursive_simpson(a, c, fa, fc, fd, tol / 2) + recursive_simpson(c, b, fc, fb, fe, tol / 2)

    fa = f(a)
    fb = f(b)
    c = (a + b) / 2
    fc = f(c)
    integral = recursive_simpson(a, b, fa, fb, fc, tol)
    return integral

def adaptive_simpson(f, a, b, tol=1e-6):
    stack = [(a, b, f(a), f(b), f((a + b) / 2))]
    integral = 0.0

    while stack:
        a, b, fa, fb, fc = stack.pop()
        c = (a + b) / 2
        h = b - a
        d = (a + c) / 2
        e = (c + b) / 2
        fd = f(d)
        fe = f(e)
        sab = h * (fa + 4 * fc + fb) / 6
        sac = h * (fa + 4 * fd + fc) / 6
        scb = h * (fc + 4 * fe + fb) / 6
        sabc = sac + scb

        if abs(sabc - sab) <= 15 * tol:
            integral += sabc + (sabc - sab) / 15
        else:
            stack.extend([(a, c, fa, fc, fd), (c, b, fc, fb, fe)])

    return integral


# Test program
def test_integration_functions():
    def test_function_1(x):
        return np.exp(-x) * np.cos(x)

    def test_function_2(x):
        return 1 / (1 + x**2)

    def test_function_3(x):
        return np.sin(x)

    def test_function_4(x):
        return np.exp(x)

    def test_function_5(x):
        return np.arctan(x)

    a = 0
    b = 2 * np.pi
    n = 1000

    print(f"Analytical value for ∫₀²π e^(-x)cos(x) dx: {1/2}")
    print(f"Lower sum: {lower_sum(test_function_1, a, b, n)}")
    print(f"Upper sum: {upper_sum(test_function_1, a, b, n)}")
    print(f"Composite Trapezoid: {composite_trapezoid(test_function_1, a, b, n)}")
    print(f"Romberg: {romberg(test_function_1, a, b, n)}")
    print(f"Composite Simpson: {composite_simpson(test_function_1, a, b, n)}")
    print(f"Adaptive Simpson: {adaptive_simpson(test_function_1, a, b)}\n")

    print(f"Analytical value for ∫₀¹ (1 + x²)^(-1) dx: {np.pi / 2}")
    print(f"Lower sum: {lower_sum(test_function_2, a, b, n)}")
    print(f"Upper sum: {upper_sum(test_function_2, a, b, n)}")
    print(f"Composite Trapezoid: {composite_trapezoid(test_function_2, a, b, n)}")
    print(f"Romberg: {romberg(test_function_2, a, b, n)}")
    print(f"Composite Simpson: {composite_simpson(test_function_2, a, b, n)}")
    print(f"Adaptive Simpson: {adaptive_simpson(test_function_2, a, b)}\n")

    print(f"Analytical value for ∫₀π sin(x) dx: {2}")
    print(f"Lower sum: {lower_sum(test_function_3, a, b, n)}")
    print(f"Upper sum: {upper_sum(test_function_3, a, b, n)}")
    print(f"Composite Trapezoid: {composite_trapezoid(test_function_3, a, b, n)}")
    print(f"Romberg: {romberg(test_function_3, a, b, n)}")
    print(f"Composite Simpson: {composite_simpson(test_function_3, a, b, n)}")
    print(f"Adaptive Simpson: {adaptive_simpson(test_function_3, a, b)}\n")

    print(f"Analytical value for ∫₀¹ e^x dx: {np.exp(1) - 1}")
    print(f"Lower sum: {lower_sum(test_function_4, a, b, n)}")
    print(f"Upper sum: {upper_sum(test_function_4, a, b, n)}")
    print(f"Composite Trapezoid: {composite_trapezoid(test_function_4, a, b, n)}")
    print(f"Romberg: {romberg(test_function_4, a, b, n)}")
    print(f"Composite Simpson: {composite_simpson(test_function_4, a, b, n)}")
    print(f"Adaptive Simpson: {adaptive_simpson(test_function_4, a, b)}\n")

    print(f"Analytical value for ∫₀¹ arctan(x) dx: {np.pi / 4}")
    print(f"Lower sum: {lower_sum(test_function_5, a, b, n)}")
    print(f"Upper sum: {upper_sum(test_function_5, a, b, n)}")
    print(f"Composite Trapezoid: {composite_trapezoid(test_function_5, a, b, n)}")
    print(f"Romberg: {romberg(test_function_5, a, b, n)}")
    print(f"Composite Simpson: {composite_simpson(test_function_5, a, b, n)}")
    print(f"Adaptive Simpson: {adaptive_simpson(test_function_5, a, b)}")

if __name__ == "__main__":
    test_integration_functions()
