""""
________________________________________________________________________________________________________________________
Based on : http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
		   https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
		   http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/
To better understand how neural networks work, I should create one from scratch. Initially this tutorial doesn't
derive all functionality and takes some things for granted (data generation, as well as some of the `mathy` functions)

In time I hope to update everything that is required. I will also try to write this per google
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
		self.data = make_moons(200, noise=0.20)
		self.X = self.data[0]
		self.y = self.data[1]
		self.w1 = np.random.rand(self.X.shape[1], 4)
		self.w2 = np.random.rand(4, 1)
		self.output = np.zeros(self.y.shape)

		# unsure of best place to put this.
		# self.seed = 0
		# np.seed(self.seed)
	def calc_loss(self):
		W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
		# calculate the first layer input
		z1 = np.dot(self.X, W1)
		# activation of first input
		a1 = np.tanh(z1)
		# calculate second layer input
		z2 = np.dot(a1, W2) + b2
		# this is our yhat value (softmax(z2)
		# also, calculate the loss by finding actual values
		a2 = np.exp(z2)/float(np.sum(np.exp(z2), axis=1, ))
		actual = -np.log

	def __repr__(self):
		""" For debugging """
		return "ANN()"

	def __str__(self):
		""" For the user """
		return "Instance of {}".format(self.__repr__())



if __name__ == "__main__":
	m = ANN()
