import unittest
import numpy as np
from NN import (SigmoidActivationLayer, ReLUActivationLayer, SoftmaxLayer,
                FullyConnectedLayer, sigmoid_prime, ReLU_prime)

class TestSigmoidActivationLayer(unittest.TestCase):

    def test_forward(self):
        sigmoid_layer = SigmoidActivationLayer()
        input_data = np.array([[0.0, 1.0],
                               [-1.0, 2.0]])
        output = sigmoid_layer.forward(input_data)
        expected_output = np.array([[0.5, 0.73105858],
                                    [0.26894142, 0.88079708]])
        self.assertEqual(output.shape, input_data.shape)
        np.testing.assert_allclose(output, expected_output, rtol=1e-6)

    def test_backward(self):
        sigmoid_layer = SigmoidActivationLayer()
        input_data = np.array([[0.0, 1.0],
                               [-1.0, 2.0]])
        sigmoid_layer.forward(input_data)
        output_gradient = np.array([[0.1, 0.2],
                                    [0.3, 0.4]])
        input_gradient = sigmoid_layer.backward(output_gradient)
        expected_input_gradient = sigmoid_prime(input_data) * output_gradient
        self.assertEqual(input_gradient.shape, input_data.shape)
        np.testing.assert_allclose(input_gradient, expected_input_gradient, rtol=1e-6)


class TestReLUActivationLayer(unittest.TestCase):

    def test_forward(self):
        relu_layer = ReLUActivationLayer()
        input_data = np.array([[-1.0, 0.0],
                               [1.0, 2.0]])
        output = relu_layer.forward(input_data)
        expected_output = np.maximum(0, input_data)
        self.assertEqual(output.shape, input_data.shape)
        np.testing.assert_allclose(output, expected_output, rtol=1e-6)

    def test_backward(self):
        relu_layer = ReLUActivationLayer()
        input_data = np.array([[-1.0, 0.0],
                               [1.0, 2.0]])
        relu_layer.forward(input_data)
        output_gradient = np.array([[0.1, 0.2],
                                    [0.3, 0.4]])
        input_gradient = relu_layer.backward(output_gradient)
        expected_input_gradient = ReLU_prime(input_data) * output_gradient
        self.assertEqual(input_gradient.shape, input_data.shape)
        np.testing.assert_allclose(input_gradient, expected_input_gradient, rtol=1e-6)


class TestSoftmaxLayer(unittest.TestCase):

    def test_forward(self):
        softmax_layer = SoftmaxLayer()
        input_data = np.array([[1.0, 2.0, 3.0],
                               [0.1, 0.2, 0.7]])
        output = softmax_layer.forward(input_data)
        # Compute expected softmax output
        exp_input = np.exp(input_data)
        expected_output = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        self.assertEqual(output.shape, input_data.shape)
        np.testing.assert_allclose(output, expected_output, rtol=1e-6)

    def test_backward(self):
        softmax_layer = SoftmaxLayer()
        # Use a single sample for simplicity
        input_data = np.array([[1.0, 2.0, 3.0]])
        output = softmax_layer.forward(input_data)
        # Define a one-hot encoded true label (assume class 2 is true)
        y_true = np.array([[0, 0, 1]])
        # Backward pass for softmax with cross-entropy simplifies to (softmax output - y_true)
        grad = softmax_layer.backward(y_true)
        expected_grad = output - y_true
        self.assertEqual(grad.shape, input_data.shape)
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-6)


class TestFullyConnectedLayer(unittest.TestCase):

    def test_forward(self):
        # Create a FullyConnectedLayer with known weights and bias.
        fc_layer = FullyConnectedLayer(input_size=3, output_size=2)
        # Overwrite random weights and bias for deterministic behavior.
        fc_layer.weights = np.array([[0.1, 0.2],
                                     [0.3, 0.4],
                                     [0.5, 0.6]])
        fc_layer.bias = np.array([[0.01, 0.02]])
        input_data = np.array([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])
        output = fc_layer.forward(input_data)
        expected_output = np.dot(input_data, fc_layer.weights) + fc_layer.bias
        self.assertEqual(output.shape, (input_data.shape[0], fc_layer.weights.shape[1]))
        np.testing.assert_allclose(output, expected_output, rtol=1e-6)

    def test_backward(self):
        fc_layer = FullyConnectedLayer(input_size=3, output_size=2)
        fc_layer.weights = np.array([[0.1, 0.2],
                                     [0.3, 0.4],
                                     [0.5, 0.6]])
        fc_layer.bias = np.array([[0.01, 0.02]])
        input_data = np.array([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])
        fc_layer.forward(input_data)  # Store input internally
        output_gradient = np.array([[0.1, 0.2],
                                    [0.3, 0.4]])
        input_gradient = fc_layer.backward(output_gradient)
        # Expected input gradient: dot(output_gradient, weights.T)
        expected_input_gradient = np.dot(output_gradient, fc_layer.weights.T)
        # Expected weight gradient: dot(input_data.T, output_gradient)
        expected_weights_gradient = np.dot(input_data.T, output_gradient)
        # Expected bias gradient: sum over the samples (axis=0), keep dims.
        expected_bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        self.assertEqual(input_gradient.shape, input_data.shape)
        np.testing.assert_allclose(input_gradient, expected_input_gradient, rtol=1e-6)
        np.testing.assert_allclose(fc_layer.delta_w, expected_weights_gradient, rtol=1e-6)
        np.testing.assert_allclose(fc_layer.delta_b, expected_bias_gradient, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()