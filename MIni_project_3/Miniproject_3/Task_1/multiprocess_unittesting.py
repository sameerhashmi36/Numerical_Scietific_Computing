import unittest
from unittest.mock import patch
import numpy as np
from multiprocessing import Pool, cpu_count
from multiprocess_approach import mandelbrot, compute_mandelbrot_row, generate_mandelbrot_parallel

class TestMandelbrot(unittest.TestCase):

    def test_mandelbrot(self):
        # Test case 1: Testing a point known to be in the Mandelbrot set
        c1 = complex(-0.5, 0)
        max_iterations1 = 100
        self.assertEqual(mandelbrot(c1, max_iterations1), max_iterations1)

        # Test case 2: Testing a point known to escape the Mandelbrot set quickly
        c2 = complex(2, 0)
        max_iterations2 = 100
        self.assertEqual(mandelbrot(c2, max_iterations2), 2)

        # Test case 3: Testing a point close to the boundary of the Mandelbrot set
        c3 = complex(0, 0.5)
        max_iterations3 = 100
        self.assertEqual(mandelbrot(c3, max_iterations3), max_iterations3)

    @patch('multiprocess_approach.cpu_count', return_value=2)
    @patch('multiprocess_approach.Pool')
    def test_generate_mandelbrot_parallel(self, mock_pool, mock_cpu_count):
        width, height = 3, 3
        x_min, x_max = -2.0, 1.0
        y_min, y_max = -1.5, 1.5
        max_iterations = 100
        generate_mandelbrot_parallel(width, height, x_min, x_max, y_min, y_max, max_iterations)
        mock_cpu_count.assert_called_once()
        mock_pool.assert_called_once_with(processes=2)

if __name__ == '__main__':
    unittest.main()
