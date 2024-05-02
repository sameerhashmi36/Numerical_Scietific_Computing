import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy_approach import mandelbrot, generate_mandelbrot

class TestMandelbrotFunctions(unittest.TestCase):

    def test_mandelbrot(self):
        # Test cases for mandelbrot function
        self.assertEqual(mandelbrot(1+1j, 100), 2)
        self.assertEqual(mandelbrot(0, 100), 100)
        self.assertEqual(mandelbrot(-1-1j, 100), 3)

    def test_generate_mandelbrot(self):
        # Test cases for generate_mandelbrot function
        mandelbrot_set = generate_mandelbrot(3, 3, -2, 1, -1, 1, 10)
        expected_set = np.array([[1., 4., 2.], [10., 10., 3.], [1., 4., 2.]])
        assert_array_almost_equal(mandelbrot_set, expected_set)

if __name__ == '__main__':
    unittest.main()
