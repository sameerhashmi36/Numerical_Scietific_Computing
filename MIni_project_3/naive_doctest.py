import doctest

def mandelbrot(c, max_iterations):

    """
        Compute the Mandelbrot iteration count for a given complex number.

        Parameters:
            c (complex): The complex number for which Mandelbrot iteration count is to be computed.
            max_iterations (int): The maximum number of iterations to perform.

        Returns:
            int: The number of iterations taken to escape the Mandelbrot set for the given complex number.

        >>> mandelbrot(1+1j, 100)
        2
        >>> mandelbrot(0, 100)
        100
        >>> mandelbrot(-1-1j, 100)
        3
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

        >>> generate_mandelbrot(3, 3, -2, 1, -1, 1, 10)
        [[1, 3, 10], [1, 10, 10], [1, 10, 10]]
        >>> generate_mandelbrot(2, 2, -2, 2, -2, 2, 1)
        [[1, 1], [1, 1]]
        >>> generate_mandelbrot(2, 2, -1, 1, -1, 1, 2)
        [[2, 2], [2, 2]]
    """

    mandelbrot_set = []

    real_step = (xmax - xmin) / width
    imag_step = (ymax - ymin) / height

    for i in range(height):

        row = []

        for j in range(width):

            real = xmin + j * real_step
            imaginary = ymin + i * imag_step
            c = complex(real, imaginary)

            row.append(mandelbrot(c, max_iterations))
        
        mandelbrot_set.append(row)

    return mandelbrot_set

if __name__ == "__main__":

    doctest.testmod(verbose=True)