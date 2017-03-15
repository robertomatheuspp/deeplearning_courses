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

		self.model: represents keras' model.
		self.heatmap represents if the current object can instantiate a HeatConvNet object from itself.

	'''

	def __init__(self, weights_path=None, heatmap=False):
		self.canHeatMap = False
		self.weights_path = weights_path
		
	def getModel(self):
		return self.model
	def compile(self):
		'''
			Definition of model default compilation
		'''
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(optimizer=sgd, loss='mse')
	def generateHeatConvNet(self):
		'''
			Instantiate a HeatConvNet model of the current CNN. 
			This feature must be implemented in the constructor of children classes by changing their input to (dimension, None, None)
			the classification method to a Convolutional based one.
			
			Considering that the children class allows to create a HeatConvNet, this method 
			transfer CNN object's learnt weights to the new HeatConvNet object.
			
			Returns an instantiation of HeatConvNet to the current class.
		'''
		if not self.canHeatMap:
			return None
		convnet_heatmap = self.__class__(heatmap=True)
		model_heatmap = convnet_heatmap.getModel()

		#transfering the learnt weights to heatmap ConvNet
		for layer in model_heatmap.layers:
		    if layer.name.startswith("conv"):
		        orig_layer = self.model.get_layer(layer.name)
		        layer.set_weights(orig_layer.get_weights())
		    elif layer.name.startswith("dense"):
		        orig_layer = self.model.get_layer(layer.name)
		        W,b = orig_layer.get_weights()
		        n_filter,previous_filter,ax1,ax2 = layer.get_weights()[0].shape
		        new_W = W.reshape((previous_filter,ax1,ax2,n_filter))
		        new_W = new_W.transpose((3,0,1,2))
		        new_W = new_W[:,:,::-1,::-1]
		        layer.set_weights([new_W,b])

		return convnet_heatmap

class HeatConvNet(CNN):
	'''	
		Base class to be used on every Heat Map Generation Convolutional Neural Network. 

	'''
	def __init__(self):
		CNN.__init__(self)
		model_heat = self.getModel()
		model_heat.compile(optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
		                    loss='mse')

	def getHeatMap(self, inputImg):
		'''
			Given an inputImg it returns a heat map image.
			This image represents in what area the given HeatConvNet find an specific class.

			inputImg is the input data, as a Numpy array.

			Returns the heat map of every class. To get a specific class' heat map use result[id_class]
			or if you are using a set of classes you might do result[ [id_1, ... id_n ] ].sum(axis=0)
		'''
		model_heat = self.getModel()		
		out_img = model_heat.predict(inputImg)

		return out_img[0]
  	