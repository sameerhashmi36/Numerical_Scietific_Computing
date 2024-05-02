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

def generate_mandelbrot_parallel(args):
    num_processes, chunk_size = args
    pool = Pool(processes=num_processes)
    args_list = [(i, width, height, x_min, x_max, y_min, y_max, max_iterations) for i in range(height)]
    mandelbrot_rows = pool.map(compute_mandelbrot_row, args_list, chunksize=chunk_size)
    pool.close()
    pool.join()
    mandelbrot_set = np.array(mandelbrot_rows)
    return mandelbrot_set

if __name__ == "__main__":
    width, height = 1000, 1000
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    max_iterations = 100
    num_processes_list = [1, 2, 3, 4, 5, 6, 7, 8]
    chunk_size_list = [10, 50, 100, 200, 500, 1000]

    execution_times = {num_processes: [] for num_processes in num_processes_list}

    for num_processes in num_processes_list:
        for chunk_size in chunk_size_list:
            start_time = time.time()
            _ = generate_mandelbrot_parallel((num_processes, chunk_size))
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times[num_processes].append(execution_time)
            print(f"Processes: {num_processes}, Chunk size: {chunk_size}, Execution time: {execution_time:.2f} seconds")

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    for num_processes in num_processes_list:
        plt.plot(chunk_size_list, execution_times[num_processes], marker='o', label=f"{num_processes} Processes")

    plt.title('Chunk Size vs Execution Time for Different Numbers of Processes')
    plt.xlabel('Chunk Size')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig('./chunk_size_vs_execution_time.png')
    plt.show()
