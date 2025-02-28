import unittest
import numpy as np
from NN import sigmoid, sigmoid_prime, ReLU, ReLU_prime

class TestActivations(unittest.TestCase):

    def test_sigmoid(self):
        x = np.array([[0, 1], [-1, 100]])
        expected = np.array([[0.5, 0.73105858], [0.26894142, 1.0]])
        np.testing.assert_allclose(sigmoid(x), expected, rtol=1e-6)

    def test_sigmoid_prime(self):
        x = np.array([[0, 1], [-1, 100]])
        expected = np.array([[0.25, 0.19661193], [0.19661193, 0.0]])
        np.testing.assert_allclose(sigmoid_prime(x), expected, rtol=1e-6)

    def test_ReLU(self):
        x = np.array([[-1, 0], [1, -100]])
        expected = np.array([[0, 0], [1, 0]])
        np.testing.assert_equal(ReLU(x), expected)

    def test_ReLU_prime(self):
        x = np.array([[-1, 0], [1, -100]])
        expected = np.array([[0, 0], [1, 0]])
        np.testing.assert_equal(ReLU_prime(x), expected)

if __name__ == '__main__':
    unittest.main()