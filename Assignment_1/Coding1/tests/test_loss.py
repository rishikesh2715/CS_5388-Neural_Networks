import unittest
import numpy as np
from NN import cross_entropy

class TestLoss(unittest.TestCase):

    def test_cross_entropy(self):
        # Predicted probabilities for 3 samples and 3 classes.
        y_pred = np.array([
            [0.2, 0.5, 0.3],  # sample 1
            [0.5, 0.1, 0.4],  # sample 2
            [0.3, 0.3, 0.4]   # sample 3
        ])
        # True class indices.
        y = np.array([1, 2, 0])
        
        # Convert labels to one-hot encoding.
        y_onehot = np.zeros_like(y_pred)
        y_onehot[np.arange(len(y)), y] = 1

        # Expected loss computed as:
        # loss = ( -ln(0.5) - ln(0.4) - ln(0.3) ) / 3 ≈ 0.937803
        expected_loss = 0.937803

        loss = cross_entropy(y_onehot, y_pred)
        self.assertAlmostEqual(loss, expected_loss, places=5)

if __name__ == '__main__':
    unittest.main()