""""
________________________________________________________________________________________________________________________
Based on : http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
		   https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
		   http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/
To better understand how neural networks work, I should create one from scratch. Initially this tutorial doesn't
derive all functionality and takes some things for granted (data generation, as well as some of the `mathy` functions)
but I hope to implement those myself in the future
________________________________________________________________________________________________________________________
"""

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

#

class NeuralNetwork(object):
	"""The model used to generate the artificial neural network

	It will create a neural network that can identify """
	def __init__(self):
		self.model = None
		self.data = make_moons(200, noise=0.20)
		self.X = self.data[0]
		self.y = self.data[1]
		self.w1 = np.random.rand(self.X.shape[1], 4)
		self.w2 = np.random.rand(4, 1)
		self.output = np.zeros(self.y.shape)
		# gradient descent parameters
		self.regular = .01 # regularization strength
		self.eps = .01 # learning rate for gradient descent

		# unsure of best place to put this.
		# self.seed = 0
		# np.seed(self.seed)

	# This function learns parameters for the neural network and returns the model.
	# - nn_hdim: Number of nodes in the hidden layer
	# - num_passes: Number of passes through the training data for gradient descent
	# - print_loss: If True, print the loss every 1000 iterations
	def build_model(self, nodes, num_passes=20000, print_loss=False):

		# Initialize the parameters to random values. We need to learn these.
		np.random.seed(0)
		W1 = np.random.randn(2, nodes) / np.sqrt(2)
		b1 = np.zeros((1, nodes))
		W2 = np.random.randn(nodes, 2) / np.sqrt(nodes)
		b2 = np.zeros((1, 2))

		# This is what we return at the end
		model = {}

		# Gradient descent. For each batch...
		for i in range(0, num_passes):

			# Forward propagation
			z1 = X.dot(W1) + b1
			a1 = np.tanh(z1)
			z2 = a1.dot(W2) + b2
			exp_scores = np.exp(z2)
			probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

			# Backpropagation
			delta3 = probs
			delta3[range(num_examples), y] -= 1
			dW2 = (a1.T).dot(delta3)
			db2 = np.sum(delta3, axis=0, keepdims=True)
			delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
			dW1 = np.dot(X.T, delta2)
			db1 = np.sum(delta2, axis=0)

			# Add regularization terms (b1 and b2 don't have regularization terms)
			dW2 += self.regular * W2
			dW1 += self.regular * W1

			# Gradient descent parameter update
			W1 += -self.eps * dW1
			b1 += -self.eps * db1
			W2 += -self.eps * dW2
			b2 += -self.eps * db2

			# Assign new parameters to the model
			model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

			# Optionally print the loss.
			# This is expensive because it uses the whole dataset, so we don't want to do it too often.
			if print_loss and i % 1000 == 0:
				print
				"Loss after iteration {} {}".format(i, self._calc_loss(model))

		return model

	def _softmax(self, input_value):
		"""
		This is a softmax helper function
		:return: float
		"""
		return np.exp(input_value)/float(np.sum(np.exp(input_value), axis=1, ))

	def _calc_loss(self):
		W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
		# calculate the first layer input
		z1 = np.dot(self.X, W1)
		# activation of first input
		a1 = np.tanh(z1)
		# calculate second layer input
		z2 = np.dot(a1, W2) + b2
		# calculate expected scores
		exp_scores = np.exp(z2)
		# softmax outputs
		softmax_prob = self._softmax(z2)
		num_examples = len(self.X)
		# let's work on doing the actualy cross entropy formula to find our loss
		actual_scores = -np.log(softmax_prob[range(num_examples), self.y])
		loss = np.sum(actual_scores)

		# Add regularization term to loss
		loss += self.regular /2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
		return 1./num_examples*loss

	# Utility function to predict an output (0 or 1)
	def predict(self, x):
		W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
		# Forward propagation
		z1 = x.dot(W1) + b1
		# activation of first input
		a1 = np.tanh(z1)
		# second layer input
		z2 = a1.dot(W2) + b2
		# softmax outputs
		softmax_prob = self._softmax(z2)
		# returns indices of max value
		return np.argmax(softmax_prob, axis=1)

	def line_graph(self, x, y, x_title, y_title):
		"""Helper matplotlib function to plot value"""
		plt.plot(x, y)
		plt.xlabel(x_title)
		plt.ylabel(y_title)
		plt.show()


	def __repr__(self):
		""" For debugging """
		return "ANN()"

	def __str__(self):
		""" For the user """
		return "Instance of {}".format(self.__repr__())



if __name__ == "__main__":
	m = NeuralNetwork()
