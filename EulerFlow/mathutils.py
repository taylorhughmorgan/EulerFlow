"""
Author: Hugh Morgan
Date: 2025-04-08
Description: math utilities used commonly among PDE solvers
"""
import numpy as np

def hyperbolic_step(grid: np.array, offset: float, delta: float=1.0):
    """ hyperbolic tangent function to emulate a step function """
    return 0.5 * (1.0 - np.tanh( (grid - offset) / delta) )

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test_grid = np.linspace(0, 10, 300)
    test_field = hyperbolic_step(test_grid, 5.0, delta=0.25)
    plt.plot(test_grid, test_field)
    plt.grid(True)