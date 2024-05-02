import numpy as np
from numba import jit
import pytest

@jit(nopython=True)
def mandelbrot(c, max_iterations):
    
    """
        Calculate the number of iterations required for a complex number 'c' to escape the Mandelbrot set.

        Parameters:
            c (complex): The complex number for which the Mandelbrot set iteration is performed.
            max_iterations (int): The maximum number of iterations allowed.

        Returns:
            int: The number of iterations taken for the complex number 'c' to escape the Mandelbrot set,
                or 'max_iterations' if it does not escape within the maximum allowed iterations.
    """
    
    z = 0
    for n in range(max_iterations):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iterations

@jit(nopython=True)
def generate_mandelbrot(width, height, xmin, xmax, ymin, ymax, max_iterations):
   
    """
        Generate the Mandelbrot set for a given range of complex numbers.

        Parameters:
            width (int): The width of the output array (number of columns).
            height (int): The height of the output array (number of rows).
            xmin (float): The minimum value of the real part of the complex numbers.
            xmax (float): The maximum value of the real part of the complex numbers.
            ymin (float): The minimum value of the imaginary part of the complex numbers.
            ymax (float): The maximum value of the imaginary part of the complex numbers.
            max_iterations (int): The maximum number of iterations for each complex number.

        Returns:
            numpy.ndarray: A 2D array representing the Mandelbrot set, where each element 
                        represents the number of iterations taken for the corresponding 
                        complex number to escape the Mandelbrot set.
    """
    
    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)

    mandelbrot_set = np.zeros((height, width), dtype=np.int64)

    for i in range(height):
        for j in range(width):
            c = complex(real[j], imag[i])
            mandelbrot_set[i, j] = mandelbrot(c, max_iterations)

    return mandelbrot_set

def test_mandelbrot():
    # Test case 1: Testing a point known to be in the Mandelbrot set
    c1 = complex(-0.5, 0)
    max_iterations1 = 100
    assert mandelbrot(c1, max_iterations1) == max_iterations1

    # Test case 2: Testing a point known to escape the Mandelbrot set quickly
    c2 = complex(2, 0)
    max_iterations2 = 100
    assert mandelbrot(c2, max_iterations2) == 2

    # Test case 3: Testing a point close to the boundary of the Mandelbrot set
    c3 = complex(0, 0.5)
    max_iterations3 = 100
    assert mandelbrot(c3, max_iterations3) == max_iterations3

if __name__ == "__main__":
    pytest.main()
