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

    mandelbrot_set = []

    real_step = (xmax - xmin) / width
    imag_step = (ymax - ymin) / height

    for i in range(width):

        row = []

        for j in range(width):

            real = xmin + j * real_step
            imaginary = ymin + i * imag_step
            c = complex(real, imaginary)

            row.append(mandelbrot(c, max_iterations))
        
        mandelbrot_set.append(row)

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
plt.imshow(mandelbrot_set, extent=(x_min, x_max, y_min, y_max), cmap='hot', origin='lower')
plt.colorbar(label='Iteration count')
plt.title(f'Naive Mandelbrot Set (Generated in {execution_time:.2f} seconds)')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()