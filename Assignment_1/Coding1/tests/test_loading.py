import unittest
import numpy as np
from utils import load_mnist_train, load_mnist_test


class TestLoading(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def test_load_mnist(self):
        train_data, train_label, val_data, val_label = load_mnist_train()

        # Check that train_data and val_data are numpy ndarrays.
        self.assertIsInstance(train_data, np.ndarray)
        self.assertIsInstance(val_data, np.ndarray)
        self.assertIsInstance(train_label, np.ndarray)
        self.assertIsInstance(val_label, np.ndarray)

        # Ensure the number of samples is consistent.
        self.assertEqual(train_data.shape[0], train_label.shape[0])
        self.assertEqual(val_data.shape[0], val_label.shape[0])
        self.assertEqual(train_data.shape[0], 4 * val_data.shape[0])

        # Check that each image has 784 features.
        self.assertEqual(train_data.shape[1], 784)
        self.assertEqual(val_data.shape[1], 784)

        # Check that labels are one-hot encoded (assume 10 classes for MNIST).
        self.assertEqual(train_label.shape[1], 10)
        self.assertEqual(val_label.shape[1], 10)

        # Verify that each one-hot encoded label sums to 1.
        self.assertTrue(np.all(np.sum(train_label, axis=1) == 1),
                        "Each training label should have exactly one '1'.")
        self.assertTrue(np.all(np.sum(val_label, axis=1) == 1),
                        "Each validation label should have exactly one '1'.")


if __name__ == '__main__':
    unittest.main()