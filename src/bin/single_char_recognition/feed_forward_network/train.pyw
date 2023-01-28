import numpy as np
from classes.ann import Ann
from classes.mnist import Mnist
from classes.network import NeuralNetwork

# Training parameters
network_dimens = [784, 100, 100, 50, 50, 36] # IMPORTANT: first lyer must be 784, and last layer must be 36!
network_dimens_string = "4h1001005050" # Describes network dimensions: "<number of hidden layers> h <hidden layer 1 dimens> <hidden layer 2 dimens> ... <hidden layer n dimens>"
epochs = 50 # Number of training epochs
learning_rate = 0.7 # Initial learning rate of the network

training_set, test_set = Mnist.unpickle("../../../res/datasets/emnist_balanced/emnist_balanced.pkl")
network = NeuralNetwork(network_dimens)
network.sgd(training_set, epochs, 10, learning_rate, test_set)
Ann.pickle(network, "../../../res/feed_forward_network/trained_networks/ann_%s_%s.pkl" % (network_dimens_string, network.evaluate(test_set)*100/14400))