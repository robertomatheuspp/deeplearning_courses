'''
	Author: Roberto Matheus Pinheiro Pereira
	Minicurso: Aprendizagem Profunda aplicada a visão - JIM (2016)
	Developer at NCA - Nucleo de Computacao Aplicada 
	Date: 27/10/2016
	Available at: https://github.com/robertomatheuspp/JIM_2016_CNN/
'''

from keras.optimizers import SGD

class CNN (object):
	'''	
		Base class to be used on every Convolutional Neural Network. 
		It defines a default compile structure.

		self.model: represents the keras' model.
	'''
	def __init__(self):
		self.model = None
		pass
	def getModel(self):
		return self.model
	def compile(self):
		'''
			Definition of model default compilation
		'''
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(optimizer=sgd, loss='mse')