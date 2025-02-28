
"""
Neural Network Module. (c) 2024 Texas Tech University

Copyright 2024, Texas Tech University
Lubbock, Texas 79409
All Rights Reserved

Template code for CS 5388 Neural Networks

Texas Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab. This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 5388 Neural Networks is
prohibited and subject to being investigated as a violation of the Texas Tech
Student Code of Conduct.

-----do not edit anything above this line---
"""

import numpy as np

class SigmoidActivationLayer:
    """
    Sigmoid Activation Layer.

    This layer applies the sigmoid activation function elementwise to its input.
    """

    def forward(self, input_data):
        """
        Perform the forward pass using the sigmoid activation function.

        :param input_data: Input data to the layer.
        :return: Activated output after applying the sigmoid function.
        """
        # Store the input for use in the backward pass
        self.input = input_data

        #############################################################################
        # TODO:                                                                     #       
        #    1) Implement the forward process:                                      #
        #       Call sigmoid function and return its output                         #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    
    def backward(self, output_gradient):
        """
        Update the layer's parameters.

        Note: Sigmoid activation layers have no parameters to update.
        
        :param learning_rate: Learning rate (not used).
        """
        #############################################################################
        # TODO:                                                                     #       
        #    1) Implement the backward process:                                     #
        #       Call sigmoid_prime function and                                     #
        #       return its output times the output_gradient                         #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def step(self, learning_rate):
       """
        Update the layer's parameters.

        Note: Sigmoid activation layers have no parameters to update.
        
        :param learning_rate: Learning rate (not used).
        """
       return
    
    
class ReLUActivationLayer:
    """
    ReLU Activation Layer.

    This layer applies the Rectified Linear Unit (ReLU) activation function
    elementwise to its input.
    """
    
    def forward(self, input_data):
        """
        Perform the forward pass using the ReLU activation function.

        :param input_data: Input data to the layer.
        :return: Activated output after applying ReLU.
        """
        # Store the input for use in the backward pass
        self.input = input_data

        #############################################################################
        # TODO:                                                                     #       
        #    1) Implement the forward process:                                      #
        #       Call ReLU function and return its output                         #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
    
    def backward(self, output_gradient):
        """
        Perform the backward pass, computing the gradient of the loss with respect
        to the input of this layer.

        :param output_gradient: Gradient of the loss with respect to the output.
        :return: Gradient of the loss with respect to the input.
        """        
        
        #############################################################################
        # TODO:                                                                     #       
        #    1) Implement the backward process:                                     #
        #       Call ReLU_prime function and                                        #
        #       return its output times the output_gradient                         #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def step(self, learning_rate):
       """
        Update the layer's parameters.

        Note: ReLU activation layers have no parameters to update.
        
        :param learning_rate: Learning rate (not used).
        """
       return
    

class SoftmaxLayer:
    """
    Softmax Layer.

    This layer computes the softmax activation, which converts raw scores
    into probabilities.
    """
    
    def forward(self, input_data):
        """
        Perform the forward pass by computing softmax probabilities.

        :param input_data: Raw scores from the previous layer (N, num_classes).
        :return: Softmax probabilities (N, num_classes).
        """
        # Store the input for use in the backward pass
        self.input = input_data

        #############################################################################
        # TODO:                                                                     #       
        #    1) Implement the forward process:                                      #
        #       Compute the softmax of each value with respect to its row           #
        #       and return the softmax output                                       #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
    
    def backward(self, y_true):
        """
        Perform the backward pass for softmax, assuming cross-entropy loss.

        For softmax combined with cross-entropy loss, the gradient simplifies to:
        (softmax_output - y_true)

        :param y_true: True labels (one-hot encoded).
        :return: Gradient of the loss with respect to the input of the softmax layer.
        """        
        #############################################################################
        # TODO:                                                                     #       
        #    1) Implement the backward process:                                     #
        #       Compute the gradient for softmax with cross-entropy loss            #
        #       and return the gradient                                             #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
    
    def step(self, learning_rate):
       """
        Update the layer's parameters.

        Note: Softmax layers have no parameters to update.
        
        :param learning_rate: Learning rate (not used).
        """
       return


class FullyConnectedLayer:
    """
    Fully Connected (Dense) Layer.

    This layer performs a linear transformation on its input using weights
    and biases.
    """
    
    def __init__(self, input_size, output_size):
        """
        Initialize the FullyConnectedLayer with random weights and zero biases.

        :param input_size: Number of input features.
        :param output_size: Number of output neurons.
        """
        self.delta_w = np.zeros((input_size, output_size))
        self.delta_b = np.zeros((1,output_size))
        self.weights = 0.001 * np.random.randn(input_size, output_size)
        self.bias = 0.001 * np.random.randn(1, output_size)
        
    def forward(self, input_data):
        """
        Perform the forward pass of the fully connected layer.

        :param input_data: Input data (batch_size, input_size).
        :return: Output of the layer (batch_size, output_size).
        """
        # Store the input for use in the backward pass
        self.input = input_data

        #############################################################################
        # TODO:                                                                     #       
        #    1) Implement the forward process                                       #                                       #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
    
    def backward(self, output_gradient):
        """
        Perform the backward pass, computing gradients for weights, biases, and input.

        :param output_gradient: Gradient of the loss with respect to the output.
        :return: Gradient of the loss with respect to the input.
        """        
        #############################################################################
        # TODO:                                                                     #       
        #    1) Implement the backward process:                                     #
        #       a) Compute the gradient for the layer with respect to W and b and   #
        #          update the instance variables delta_w and delta_b                #
        #       b) Compute the gradient for the input of the layer and return it    #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def step(self, learning_rate):
        """
        Update the weights and biases using the accumulated gradients.

        :param learning_rate: Learning rate for the parameter update.
        """
        self.weights -= learning_rate * self.delta_w
        self.bias -= learning_rate * self.delta_b
        self.delta_w = np.zeros(self.weights.shape)
        self.delta_b = np.zeros(self.bias.shape)


class NeuralNetwork:
    """
    Neural Network.

    This class represents a neural network as a sequence of layers. It provides
    methods for adding layers, making predictions, computing accuracy, and training
    the network using gradient descent.
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the NeuralNetwork.

        :param verbose: If True, print training progress.
        """
        self.verbose = verbose
        self.layers = []
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def add(self, layer):
        """
        Add a layer to the neural network.

        :param layer: An instance of a layer (e.g., FullyConnectedLayer, ActivationLayer).
        """
        self.layers.append(layer)

    def predict(self, input_data):
        """
        Predict the class labels for the given input data.

        :param input_data: Input data for prediction.
        :return: Array of predicted class labels.
        """
        result = []
        for i in range(input_data.shape[0]):
            output = input_data[i:i+1]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(np.argmax(output))
        return np.array(result)
    
    def compute_accuracy(self, x, y):
        """
        Compute the accuracy of the network's predictions.

        :param x: Input data.
        :param y: True labels (one-hot encoded).
        :return: Accuracy as a float.
        """
        predictions = self.predict(x)
        return np.mean(predictions == np.argmax(y, axis=1))
    
    def fit(self, x_train, y_train, x_val, y_val, epochs=1, learning_rate=1.0, batch_size=64):
        """
        Train the neural network using the provided training data and validate on the
        validation data.

        :param x_train: Training data inputs.
        :param y_train: Training data true labels (one-hot encoded).
        :param x_val: Validation data inputs.
        :param y_val: Validation data true labels (one-hot encoded).
        :param epochs: Number of training epochs.
        :param learning_rate: Learning rate for weight updates.
        :param batch_size: Batch size for mini-batch gradient descent.
        """
        n = x_train.shape[0]
        for epoch in range(epochs):
            train_loss_sum = 0  # Track total training loss
            
            # Shuffle the training data at the beginning of each epoch
            indices = np.arange(n)
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

            for i in range(0, n, batch_size):
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                # Forward pass:
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)

                # Compute loss
                train_loss_sum += cross_entropy(y_batch, output)

                # Backward pass:
                # Compute gradients for the last layer
                gradients = self.layers[-1].backward(y_batch)
                for layer in reversed(self.layers[:-1]):
                    gradients = layer.backward(gradients)

                # Update all layers
                for layer in self.layers:
                    layer.step(learning_rate)

            # Compute average training loss
            train_loss_avg = train_loss_sum / (n // batch_size)

            # Compute training accuracy
            train_accuracy = self.compute_accuracy(x_train, y_train)

            # Compute validation loss
            val_loss_sum = 0
            for i in range(x_val.shape[0]):
                output = x_val[i:i+1]
                for layer in self.layers:
                    output = layer.forward(output)
                val_loss_sum += cross_entropy(y_val[i], output)
            val_loss_avg = val_loss_sum / x_val.shape[0]

            # Compute validation accuracy
            val_accuracy = self.compute_accuracy(x_val, y_val)

            # Store loss and accuracy for plotting
            self.train_loss_history.append(train_loss_avg)
            self.train_acc_history.append(train_accuracy)
            self.val_loss_history.append(val_loss_avg)
            self.val_acc_history.append(val_accuracy)

            if self.verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                    f"TrainLoss: {train_loss_avg:.4f}, TrainAcc: {train_accuracy:.4f}, "
                    f"ValLoss: {val_loss_avg:.4f}, ValAcc: {val_accuracy:.4f}")



def sigmoid(x):
    """
    Compute the sigmoid activation function.

    :param x: Input data.
    :return: Output after applying the sigmoid function.
    """
    out = None
    #############################################################################
    # TODO: Compute the sigmoid activation on the input x                       #
    #############################################################################

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################


def sigmoid_prime(x):
    """
    Compute the derivative of the sigmoid function.

    :param x: Input data.
    :return: Derivative of the sigmoid function.
    """
    ds = None
    #############################################################################
    # TODO:                                                                     #
    #    1) Implement the derivative of Sigmoid function                        #
    #############################################################################

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return ds

def ReLU(x):
    """
    Compute the Rectified Linear Unit (ReLU) activation.

    :param x: Input data.
    :return: Output after applying ReLU (elementwise maximum of 0 and x).
    """
    out = None
    #############################################################################
    # TODO: Compute the ReLU activation on the input                            #
    #############################################################################

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return out


def ReLU_prime(x):
    """
    Compute the derivative of the ReLU activation.

    :param x: Input data.
    :return: Derivative of ReLU (1 if x > 0, else 0).
    """
    dr = None
    #############################################################################
    # TODO: Compute the gradient of ReLU activation                             #
    #############################################################################

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return dr


def cross_entropy(y, y_pred):
    """
    Compute the cross-entropy loss between the true labels and predicted probabilities.

    :param y: True labels (one-hot encoded).
    :param y_pred: Predicted probabilities.
    :return: Cross-entropy loss.
    """
    loss = None
    #############################################################################
    # TODO:                                                                     #
    #    1) Implement Cross-Entropy Loss                                        #
    #############################################################################

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return loss


