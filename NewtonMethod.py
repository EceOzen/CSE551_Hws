# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:42:20 2023

@author: Ece
"""

import math

def newton(f, f_prime, x0, tol=1e-6, max_iter=100, m=1):
    """
    Newton's method for finding roots with multiplicity.

    Parameters:
    - f: The function for which roots are to be found.
    - f_prime: The derivative of the function f.
    - x0: Initial guess for the root.
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.
    - m: Multiplicity of the root (default value is 1).

    Returns:
    - root: The estimated root.
    - iterations: Number of iterations performed.
    """

    x = x0
    iterations = 0

    while iterations < max_iter:
        fx = f(x)
        fpx = f_prime(x)

        if abs(fx) < tol:
            # If the function value is already close to zero, we consider it converged.
            return x, iterations

        if abs(fpx) < tol:
            # If the derivative is too small, Newton's method may not converge well.
            raise ValueError("Derivative is too small. Choose a different initial guess.")

        # Newton's method update with multiplicity parameter m.
        x = x - m * fx / fpx
        iterations += 1

        if abs(fx) < tol:
            # Check for convergence after the update.
            return x, iterations

    raise ValueError("Newton's method did not converge within the maximum number of iterations.")
    
def test_Newton():
    def f1(x):
        return x**2 

    def f1_prime(x):
        return 2*x

    def f(x):
        return x**2 - 4

    def g(x):
        fx = f(x)
        if fx == 0:
            raise ValueError("ZeroDivisionError: f(x) is zero. Choose a different initial guess.")
        return (f(x + fx) - fx) / fx

    print("Solving CP 3.2.1:")
    try:
        root_1, iterations_1 = newton(f1, f1_prime, x0=1)
        print(f"Root 1: {root_1}, Iterations: {iterations_1}")
    except ValueError as e:
        print(f"Error: {e}")

    initial_guesses = [1, -1, 3]  # Try different initial guesses
    print("\nSolving CP 3.2.36:")
    for guess in initial_guesses:
        try:    
            print(f"Initial guess is {guess}")
            root_2, iterations_2 = newton(f, g, x0=guess)
            print(f"Root 2: {root_2}, Iterations: {iterations_2}")
        except ValueError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_Newton()
