# Creating a neural network from scratch
# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
# Components
# Input layer, x
# No. hidden layers
# output layer y^
# set of weights and biases between each layer, W and b
# activation function for each hidden layer (sigma) -> sigmoid here

import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    # assuming 1 layer NN
    f = sigmoid(x)
    ds = f * (1 - f)
    return ds


class NeuralNetwork:
    # creating the neural network shape
    # __init__ - object created from a class, allows the class to init the attrs of class
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    # training the neural network
    # 1 - calculating the pred y, feedforward
    # 2 - updating weights and biases, backpropogation
    # for simplicity assuming biases to be 0
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    # Backpropogation
    #
    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                  self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
