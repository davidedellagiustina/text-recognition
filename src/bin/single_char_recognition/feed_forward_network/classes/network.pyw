import numpy as np

# This class represents the whole neural network

class NeuralNetwork(object):

	# Constructor - initializes a network with the given dimensions (dimens must be a list of the neurons contained in each layer)

	def __init__(self, dimens):
		self.num_layers = len(dimens)
		self.dimens = dimens
		self.weights = [np.random.randn(y, x) for x, y in zip(self.dimens[:-1], self.dimens[1:])]
		self.biases = [np.random.randn(y, 1) for y in self.dimens[1:]]

	# Function that, given an input "a" for the network, returns the corresponding output

	def feedforward(self, a):
		for w, b in zip(self.weights, self.biases):
			a = sigmoid(np.dot(w, a) + b)
		return a

	# Function that, for every epoch, shuffles the training data and creates mini batches (according to the stochastic gradient descent algoritm) with the given size, then it calls the "update_mini_batch" function. If "test_data" is given, a test against it is done at the end of each epoch of training

	def sgd(self, training_data, epochs, mini_batch_size, eta, test_data = None):
		if test_data: n_test = len(test_data)
		n = len(training_data)
		for e in range(epochs):
			np.random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)
			if test_data:
				print("Epoch %s: %s / %s" % (e, self.evaluate(test_data), n_test))
			else:
				print("Epoch %s complete" % (e))
			eta *= 0.95

	# Function that updates weights and biases basing on the nabla_w and nabla_b vertors, which are computed by the "backprop" function

	def update_mini_batch(self, mini_batch, eta):
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		for x, y in mini_batch:
			delta_nabla_w, delta_nabla_b = self.backprop(x, y)
			nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
			nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
		self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

	# Function that calculates the gradient of the cost function (with respect to weights and biases) by using the backpropagation algorithm

	def backprop(self, x, y):
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		# Feedforward
		activation = x
		activations = [x]
		zs = []
		for w, b in zip(self.weights, self.biases):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# Backward pass
		delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		nabla_b[-1] = delta
		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
			nabla_b = delta
		return (nabla_w, nabla_b)

	# Function that returns the derivative of the cost function, given the output of the net and the expected result

	def cost_derivative(self, output_activations, y):
		return (output_activations - y)

	# Function that evaluates the neural network precision against a test dataset

	def evaluate(self, test_data):
		test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

# MISCELLANEOUS (YET USEFUL) FUNCTIONS

# Function that returns the value of the sigmoid function for a certain input value

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

# Function that returns the value of the derivative of the sigmoid function for a certain input value

def sigmoid_prime(x):
	return sigmoid(x) * (1 - sigmoid(x))