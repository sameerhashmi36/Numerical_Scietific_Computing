import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit

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

def main():
    
    """
        Main function to generate and display the Mandelbrot set using Numba-accelerated computation.
    """
    
    width, height = 1000, 1000
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    max_iterations = 100

    start_time = time.time()
    mandelbrot_set = generate_mandelbrot(width, height, x_min, x_max, y_min, y_max, max_iterations)
    end_time = time.time()
    execution_time = end_time - start_time

    plt.figure(figsize=(10, 10))
    plt.imshow(mandelbrot_set, extent=(x_min, x_max, y_min, y_max), cmap='hot', origin='lower')
    plt.colorbar(label='Iteration count')
    plt.title(f'Numpy vectorized Mandelbrot Set (Generated in {execution_time:.2f} seconds)')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()

if __name__ == "__main__":
    main()
