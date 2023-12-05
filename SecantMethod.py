# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:06:14 2023

@author: Ece
"""

def secant(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Generalized Secant method for finding roots.

    Parameters:
    - f: The function for which roots are to be found.
    - x0, x1: Initial guesses for the root.
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.

    Returns:
    - root: The estimated root.
    - iterations: Number of iterations performed.
    """

    x_prev = x0
    x_curr = x1
    iterations = 0

    while iterations < max_iter:
        fx_curr = f(x_curr)
        fx_prev = f(x_prev)

        if abs(fx_curr) < tol:
            # If the function value is already close to zero, we consider it converged.
            return x_curr, iterations

        if abs(fx_curr - fx_prev) < tol:
            # If the difference between consecutive values is too small, Secant may not converge well.
            raise ValueError("Secant method may not converge. Choose different initial guesses.")

        # Secant method update.
        x_next = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        iterations += 1

        if abs(f(x_next)) < tol:
            # Check for convergence after the update.
            return x_next, iterations

        x_prev, x_curr = x_curr, x_next

    raise ValueError("Secant method did not converge within the maximum number of iterations.")


def test_Secant():
    def f(x):
        return x**3 - 2*x + 2

    print("Solving CP 3.3.2:")
    try:
        root, iterations = secant(f, x0=0, x1=1)
        print(f"Root: {root}, Iterations: {iterations}")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_Secant()
