import numpy as np
import matplotlib.pyplot as plt
import time

def mandelbrot(c, max_iterations):
    z = 0
    for n in range(max_iterations):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iterations

def generate_mandelbrot(width, height, xmin, xmax, ymin, ymax, max_iterations):
    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)

    mandelbrot_set = np.zeros((height, width), dtype=np.float64)

    for i in range(height):
        for j in range(width):
            c = complex(real[j], imag[i])
            mandelbrot_set[i, j] = mandelbrot(c, max_iterations)

    return mandelbrot_set

width, height = 1000, 1000
x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5
max_iterations = 100

start_time = time.time()
mandelbrot_set = generate_mandelbrot(width, height, x_min, x_max, y_min, y_max, max_iterations)
end_time = time.time()
execution_time = end_time - start_time

plt.figure(figsize=(10, 10))
plt.imshow(mandelbrot_set, extent=(x_min, x_max, y_min, y_max), cmap= 'hot', origin='lower')
plt.colorbar(label='Iteration count')
plt.title(f'Numpy vectorized Mandelbrot Set (Generated in {execution_time:.2f} seconds)')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()
