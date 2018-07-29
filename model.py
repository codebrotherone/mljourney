""""
________________________________________________________________________________________________________________________
Based on : http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

To better understand how neural networks work, I should create one from scratch. Initially this tutorial doesn't
derive all functionality and takes some things for granted (data generation, as well as some of the `mathy` functions)

In time I hope to update everything that is required. I will also try to write this per google
________________________________________________________________________________________________________________________
"""

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

#

class ANN(object):
	"""The model used to generate the artificial neural network

	It will create a neural network that can identify """
	def __init__(self):
		self.data = make_moons(200, noise=0.20)
		self.X = self.data[0]
		self.y = self.data[1]
		self._single = 'single'
		self.__dunder = 'dunder'
		self.seed = 0
		np.seed(self.seed)
	
	def __repr__(self):
		""" For debugging """
		return "ANN()"

	def __str__(self):
		""" For the user """
		return "Instance of {}".format(self.__repr__())



if __name__ == "__main__":
	m = ANN()
