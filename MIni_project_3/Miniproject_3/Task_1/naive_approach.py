import matplotlib.pyplot as plt
import time

def mandelbrot(c, max_iterations):

    """
        Compute the Mandelbrot iteration count for a given complex number.

        Parameters:
            c (complex): The complex number for which Mandelbrot iteration count is to be computed.
            max_iterations (int): The maximum number of iterations to perform.

        Returns:
            int: The number of iterations taken to escape the Mandelbrot set for the given complex number.
    """
    
    z = 0

    for n in range(max_iterations):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iterations

def generate_mandelbrot(width, height, xmin, xmax, ymin, ymax, max_iterations):


    """
        Generate the Mandelbrot set as a 2D list of iteration counts for given parameters.

        Parameters:
            width (int): The width of the image (number of pixels).
            height (int): The height of the image (number of pixels).
            xmin (float): The minimum value of the real part of the complex plane.
            xmax (float): The maximum value of the real part of the complex plane.
            ymin (float): The minimum value of the imaginary part of the complex plane.
            ymax (float): The maximum value of the imaginary part of the complex plane.
            max_iterations (int): The maximum number of iterations to perform.

        Returns:
            list: A 2D list representing the Mandelbrot set, where each element is the iteration count for the corresponding complex number.
    """
    
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

def main():

    """
        Main function to generate and display the Mandelbrot set.

        This function defines the parameters for generating the Mandelbrot set, calls the
        generate_mandelbrot function to compute the Mandelbrot set, and plots the result.

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
    plt.title(f'Naive Mandelbrot Set (Generated in {execution_time:.2f} seconds)')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()

if __name__ == "__main__":
    main()
