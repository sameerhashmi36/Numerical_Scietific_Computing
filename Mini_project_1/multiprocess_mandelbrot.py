import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count

def mandelbrot(c, max_iterations):
    z = 0
    for n in range(max_iterations):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iterations

def compute_mandelbrot_row(args):
    row_idx, width, height, xmin, xmax, ymin, ymax, max_iterations = args
    row = np.zeros(width, dtype=np.int64)
    for j in range(width):
        real = xmin + j * (xmax - xmin) / (width - 1)
        imag = ymin + row_idx * (ymax - ymin) / (height - 1)
        c = complex(real, imag)
        row[j] = mandelbrot(c, max_iterations)
    return row

def generate_mandelbrot_parallel(width, height, xmin, xmax, ymin, ymax, max_iterations):
    num_processes = cpu_count()
    print(num_processes)
    pool = Pool(processes=num_processes)
    args_list = [(i, width, height, xmin, xmax, ymin, ymax, max_iterations) for i in range(height)]
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
