import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import random

class Network:
    def __init__(self, layers):

        self.num_layers = len(layers)
        self.sizes = layers
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]

    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self,z):
        return sigmoid(z)*(1-sigmoid(z))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

listw = [1,2,1,1]
network = Network([1,2,1,1])
print(network.weights)
print("------")
print(np.random.randn(1, 2))
print(np.random.randn(1, 1))
print(np.random.randn(2, 1))
