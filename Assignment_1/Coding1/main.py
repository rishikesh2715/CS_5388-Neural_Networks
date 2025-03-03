"""
Main Script. (c) 2024 Texas Tech University

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


from NN import *
from utils import plot_curves, load_mnist_test, load_mnist_train
import time

def main():
    start_time = time.time()
    x_train, y_train, x_val, y_val = load_mnist_train()
    x_test, y_test = load_mnist_test()

    # Set up parameters
    input_size = 28*28
    num_classes = 10
    batch_size = 64
    num_epochs = 10
    learning_rate = .01
    hidden_layer_size = 64

    # # Initialize model 1
    # net = NeuralNetwork()
    # net.add(FullyConnectedLayer(input_size, num_classes))
    # net.add(SigmoidActivationLayer())
    # net.add(SoftmaxLayer())
 
    # Initialize model 2
    net = NeuralNetwork()
    net.add(FullyConnectedLayer(input_size,hidden_layer_size))
    net.add(ReLUActivationLayer())
    net.add(FullyConnectedLayer(hidden_layer_size,num_classes))
    net.add(SoftmaxLayer())

    net.fit(x_train, y_train, x_val, y_val, epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size)

    training_accuracy = net.compute_accuracy(x_train, y_train)
    test_accuracy = net.compute_accuracy(x_test, y_test)
    print(f'Training Accuracy: {training_accuracy:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')

    end_time = time.time()
    print(f'Time taken: {end_time - start_time:.2f} seconds')

    plot_curves(net.train_loss_history, net.train_acc_history, net.val_loss_history, net.val_acc_history, learning_rate)

if __name__ == "__main__":
    main()

