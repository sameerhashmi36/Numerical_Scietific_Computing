import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count

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

def compute_mandelbrot_row(args):
    
    """
        Compute a single row of the Mandelbrot set.

        Parameters:
            args (tuple): A tuple containing the arguments required for computing the Mandelbrot row:
                - row_idx (int): The index of the row being computed.
                - width (int): The width of the output array (number of columns).
                - height (int): The height of the output array (number of rows).
                - xmin (float): The minimum value of the real part of the complex numbers.
                - xmax (float): The maximum value of the real part of the complex numbers.
                - ymin (float): The minimum value of the imaginary part of the complex numbers.
                - ymax (float): The maximum value of the imaginary part of the complex numbers.
                - max_iterations (int): The maximum number of iterations for each complex number.

        Returns:
            numpy.ndarray: An array representing a single row of the Mandelbrot set.
    """
    
    row_idx, width, height, xmin, xmax, ymin, ymax, max_iterations = args
    print("Arguments:")
    print("row_idx:", row_idx)
    print("width:", width)
    print("height:", height)
    print("xmin:", xmin)
    print("xmax:", xmax)
    print("ymin:", ymin)
    print("ymax:", ymax)
    print("max_iterations:", max_iterations)
    row = np.zeros(width, dtype=np.int64)
    for j in range(width):
        real = xmin + j * (xmax - xmin) / (width - 1)
        imag = ymin + row_idx * (ymax - ymin) / (height - 1)
        c = complex(real, imag)
        # print("real: " ,real)
        # print("imag: " ,imag)
        # print("c: " ,c)
        row[j] = mandelbrot(c, max_iterations)
    print("Computed Row:")
    print(row)
    return row

def generate_mandelbrot_parallel(width, height, xmin, xmax, ymin, ymax, max_iterations):
    
    """
        Generate the Mandelbrot set in parallel using multiple processes.

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
    
    num_processes = cpu_count()
    print("Number of CPU cores:", num_processes)
    pool = Pool(processes=num_processes)
    args_list = [(i, width, height, xmin, xmax, ymin, ymax, max_iterations) for i in range(height)]
    # print("Arguments list:", args_list)
    mandelbrot_rows = pool.map(compute_mandelbrot_row, args_list)
    pool.close()
    pool.join()
    mandelbrot_set = np.array(mandelbrot_rows)
    return mandelbrot_set

if __name__ == "__main__":
    width, height = 1000, 1000
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    max_iterations = 100

    start_time = time.time()
    mandelbrot_set = generate_mandelbrot_parallel(width, height, x_min, x_max, y_min, y_max, max_iterations)
    end_time = time.time()
    execution_time = end_time - start_time

    plt.figure(figsize=(10, 10))
    plt.imshow(mandelbrot_set, extent=(x_min, x_max, y_min, y_max), cmap='hot', origin='lower')
    plt.colorbar(label='Iteration count')
    plt.title(f'Parallel Mandelbrot Set (Generated in {execution_time:.2f} seconds)')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()
