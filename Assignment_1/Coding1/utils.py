"""
Utility Functions. (c) 2024 Texas Tech University

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
import matplotlib.pyplot as plt

def load_csv(path):
    """
    Load data from a CSV file where the first column is the label and the remaining columns are pixel values.

    The pixel values are normalized by dividing each value by 255.

    :param path: Path to the CSV file.
    :return: A tuple (data, labels) where:
        - data is a NumPy array of shape (num_samples, num_features) containing normalized pixel values.
        - labels is a NumPy array of shape (num_samples,) containing integer labels.
    """
    data = []
    labels = []
    with open(path, 'r') as fp:
        rows = fp.readlines()
        for row in rows:
            row = row.strip().split(',')
            data.append([int(x)/255 for x in row[1:]])
            labels.append(int(row[0]))
    return np.array(data), np.array(labels)


def load_mnist_train():
    """
    Load MNIST training data with labels
    :return:
        train_data (np.ndarray): Training images as a numpy array
        train_label (np.ndarray): One-hot encoded training labels
        val_data (np.ndarray): Validation images as a numpy array 
        val_label (np.ndarray): One-hot encoded validation labels as a numpy array 
    """
    # Load training data
    print("Loading training data...")
    data, label = load_csv('mnist_train.csv')
    assert len(data) == len(label)
    print("Training data loaded with {count} images".format(count=len(data)))

    # split training/validation data
    train_data = None
    train_label = None
    val_data = None
    val_label = None

    #############################################################################
    # TODO:                                                                     #       
    #    1) One-hot encode the labels                                           #
    #    2) Split the entire training set to training data and validation       #
    #       data. Use 80% of your data for training and 20% of your data for    #
    #       validation. Note: Don't shuffle here.                               #
    #############################################################################
    # one-shot encode 
    num_samples = len(label)
    num_classes = 10 
    one_hot_labels = np.zeros((num_samples, num_classes))

    for i in range(num_samples):
        one_hot_labels[i, label[i]] = 1
    
    # 80-20 split for training and validation data
    split_idx = int(0.8 * num_samples)
    train_data = data[:split_idx]
    train_label = one_hot_labels[:split_idx]
    val_data = data[split_idx:]
    val_label = one_hot_labels[split_idx:]

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return train_data, train_label, val_data, val_label


def load_mnist_test():
    """
        Load MNIST testing data with labels
        :return:
            test_data (np.ndarray): Testing images as a numpy array
            test_label (np.ndarray): One-hot encoded testing labels as a numpy array 
    """
    # Load testing data
    print("Loading testing data...")
    data, label = load_csv('mnist_test.csv')
    assert len(data) == len(label)
    print("Testing data loaded with {count} images".format(count=len(data)))

    #############################################################################
    # TODO:                                                                     #       
    #    1) One-hot encode the labels                                           #
    #############################################################################
    # one-hot encode the labels
    num_samples = len(label)
    num_classes = 10

    one_hot_labels = np.zeros((num_samples, num_classes))

    for i in range(num_samples):
        one_hot_labels[i, label[i]] = 1
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return data, one_hot_labels



def plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, learning_rate):
    """
    Plot learning curves with matplotlib. Make sure training loss and validation loss are plot in the same figure and
    training accuracy and validation accuracy are plot in the same figure too.
    :param train_loss_history: training loss history of epochs
    :param train_acc_history: training accuracy history of epochs
    :param valid_loss_history: validation loss history of epochs
    :param valid_acc_history: validation accuracy history of epochs
    :return: None, save two figures in the current directory
    """
    #############################################################################
    # TODO:                                                                     #
    #    1) Plot learning curves of training and validation loss                #
    #    2) Plot learning curves of training and validation accuracy            #
    #############################################################################

    epochs = range(1, len(train_loss_history) + 1)

    # plot for loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss_history, 'b-', label='Training Loss')
    plt.plot(epochs, valid_loss_history, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss with learning rate: {}'.format(learning_rate))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # plot for accuracy curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc_history, 'b-', label='Training Accuracy')
    plt.plot(epochs, valid_acc_history, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy with learning rate: {}'.format(learning_rate))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################


