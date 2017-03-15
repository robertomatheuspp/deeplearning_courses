from keras.optimizers import SGD


class CNN (object):
	def __init__(self, weights_path=None, heatmap=False):
		self.canHeatMap = False
		pass
	def getModel(self):
		return self.model
	def compile(self):
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(optimizer=sgd, loss='mse')
	def generateHeatConvNet(self):
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
	'''
	def __init__(self):
		CNN.__init__(self)

	def getHeatMap(self, inputImg):
		'''
		'''
		if not self.canHeatMap:
			return None

		model_heat = self.getModel()
		model_heat.compile(optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
		                    loss='mse')
		out_img = model_heat.predict(inputImg)

		return out_img
  	