# Do not generate "__pycache__" folder
import sys
sys.dont_write_bytecode = True

import os
import numpy as np
from joblib import dump, load
import PIL.Image, PIL.ImageFilter
from tqdm import tqdm
from multiprocessing import dummy as multiprocessing
from prettytable import PrettyTable

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from scipy import stats
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt, cm as cm
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Tensorflow
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
#import tensorflow_probability as tfp

import pyTorch as torch
import os
import torch
import torch.nn.functional as F
from joblib import dump, load
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

IMAGENET_MEAN 	= [0.485, 0.456, 0.406]
IMAGENET_STD 	= [0.229, 0.224, 0.225]

# Keras preprocessing functions
preprocess_resnet_v2 = torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
#preprocess_v3 = tf.keras.applications.inception_v3.preprocess_input
#preprocess_vgg16 = tf.keras.applications.vgg16.preprocess_input
#####
# Model class
#####

RESNET_LAYERS = ['layer1','layer2', 'layer3', 'layer4', 'avgpool', 'linear']
RESNET_LAYERS_TENSORS = {
    'layer1': (3,224,224),
    'layer2': (256,56,56),
    'layer3': (512,28,28),
    'layer4': (1024,14,14),
    'avgpool': (2048,7,7),
    'linear' : (1,2048)
}

class Model:

	##### Init #####
	def __init__(self, model_name, graph_path_filename, label_path_filename, preprocessing_function=lambda x: x / 255, binary_classification = False, max_examples=500):

		# Attributes
		self.model_name = model_name
		self.max_examples = max_examples
		self.binary_classification = binary_classification

		# Folders and directories
		self.graph_path_filename = graph_path_filename
		self.label_path_filename = label_path_filename

		self.graph_path_dir = None
		self.label_path_dir = None

		# Wrapper & preprocessing functions
		self.model_wrapper = TorchModelWrapper
		self.activation_generator = ImageActivationGenerator
		self.preprocessing_function = preprocessing_function

	##### Get layer names #####
	def getLayerNames(self):
		return [layer_name for layer_name in self.model_wrapper.layer_tensors.keys()]

	##### Print model's informations #####
	def info(self):

		# Print a table with information
		table = PrettyTable(title = f"Model: {self.model_name}", field_names=["N. classes", "Layers"], float_format='.2')
		for i, layer_name in enumerate(self.getLayerNames()):
			table.add_row([len(self.model_wrapper.labels) if i == 0 else "", layer_name])
		print(table)


#####
# TorchModelWrapper class
#####
#From Image Input to Feature Maps of a selected Layer
class FeatureMapsModel(nn.Module):
  def __init__(self, model, layer_name):
    super(FeatureMapsModel, self).__init__()
    self.model = model
    self.layer_name = layer_name

    # Crea un sottogruppo del modello fino al livello desiderato
    self.layers = nn.Sequential(*list(model.children())[:self._get_layer_index() + 1])

  def _get_layer_index(self):
    """Trova l'indice del livello specificato nel modello."""
    layer_names = [name for name, _ in self.model.named_children()]
    if self.layer_name not in layer_names:
      raise ValueError(f"Layer {self.layer_name} not found in the model.")
    return layer_names.index(self.layer_name)

  def forward(self, x):
    return self.layers(x)



#From Feature Maps of a selected Layer to Logits
#LayerName must be the next Layer from where the fmap come
class LogitsModel(nn.Module):
  def __init__(self, model, layer_name):
    super(LogitsModel, self).__init__()
    self.model = model
    self.layer_name = layer_name
    self.avg_layer = [i for i in model.modules()][-2]
    self.lin_layer = [i for i in model.modules()][-1]

    # Crea un sottogruppo del modello fino al livello desiderato
    self.conv_layers =  nn.Sequential(*list(model.children())[self._get_layer_index():-2])

  def _get_layer_index(self):
    """Trova l'indice del livello specificato nel modello."""
    layer_names = [name for name, _ in self.model.named_children()]
    if self.layer_name not in layer_names:
      raise ValueError(f"Layer {self.layer_name} not found in the model.")
    return layer_names.index(self.layer_name)

  def forward(self, x):
    if len(self.conv_layers) > 0:
      x = self.conv_layers(x)
    x = self.avg_layer(x).reshape(RESNET_LAYERS_TENSORS['linear'])
    x = self.lin_layer(x)
    return x


class PyTorchModelWrapper:
    def __init__(self, model_path, labels_path, batch_size, model_name):
	    # Model details
	    self.model_name = model_name        # Model name
	    self.layers = []                    # Layer names
	    self.layer_tensors = None           # Tensors

	    # Simulated models for specific purposes
	    self.simulated_layer_model = {}     # Simulated "layer" model
	    self.simulated_logits_model = {}    # Simulated "logits" model

	    # Batching
	    self.batch_size = batch_size

	    # Load model
	    self.model = torch.load(model_path)
	    self.model.eval()  # Set to evaluation mode

	    # Fetch layer names and tensors
	    self._get_layer_tensors()

	    # Load labels
	    with open(labels_path, 'r') as f:
	        self.labels = f.read().splitlines()


	##### Get the class label from its id #####
	def id_to_label(self, idx):
		return self.labels[idx]

	##### Get the class id from its label #####
	def label_to_id(self, label):
		return self.labels.index(label)

	##### Get the prediction(s) given one or more input(s) #####
	def get_predictions(self, imgs):
		# Convert inputs to float32 and move to device (e.g., GPU) if available
	    inputs = imgs.float().to(self.device)

	    # Set the model to evaluation mode
	    self.model.eval()

	    # Disable gradient computation for inference
	    with torch.no_grad():
	        predictions = self.model(inputs)

	    # Return the predictions
	    return predictions



	##### Get the feature maps given one or more input(s) #####

	def get_feature_maps(self, imgs, layer_name):
        # Verifica se il layer_name è valido
        if layer_name not in self.simulated_layer_model:
            self.simulated_layer_model[layer_name] = FeatureMapsModel(self.model,layer_name)

        # Assicura che le immagini siano un tensor di PyTorch
        imgs = torch.tensor(imgs, dtype=torch.float32)

        # Lista per salvare le feature maps di ogni batch
        feature_maps = []

        # Elaborazione per batch
        with torch.no_grad():  # Disattiva il calcolo del gradiente per l'inferenza
            for i in range(0, len(imgs), self.batch_size):
                # Prepara il batch corrente
                batch_imgs = imgs[i:i+self.batch_size]

                # Passa il batch attraverso il modello
                output = self.simulated_layer_model[layer_name].foward(batch_imgs)

                # Aggiunge le feature maps del layer specificato
                feature_maps.append(output.cpu().numpy())

        # Concatena le feature maps di tutti i batch
        feature_maps = np.concatenate(feature_maps, axis=0)

        # Ritorna le feature maps
        return feature_maps

	##### Get the logits given a layer and one or more input(s) #####
	def get_logits(self, feature_maps, layer_name):
		#----TO TEST----
		# Simulate a model with the logits (lazy)
		if layer_name not in self.simulated_logits_model:
			self.simulated_logits_model[layer_name] = LogitsModel(self.model,layer_name)

		# Feed the model with the inputs
		logits = self.simulated_logits_model[layer_name].forward(feature_maps)

		# Return the logits
		return logits

	##### Get the gradients given a layer and one or more input(s) #####
	def get_gradient_of_score(self, feature_maps, layer_name, target_class_index):
		#----TO TEST----
		# Simulate a model with the logits (lazy)
		if layer_name not in self.simulated_logits_model:
			self.simulated_logits_model[layer_name] = LogitsModel(self.model,layer_name,)

		# Executing the gradients computation (batching)
		gradients = []
		# Process in batches
        for i in range(0, len(feature_maps), self.batch_size):
            inputs = feature_maps[i : i + self.batch_size]
            inputs = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)

            # Forward pass to get logits and compute gradients
            logits = self.simulated_logits_model[layer_name].forward(inputs)
            logit = logits[:, target_class_index]  # Select logits for target class

            # Compute gradients
            logit_sum = logit.sum()  # Sum for batch processing
            logit_sum.backward()

            # Retrieve gradients for inputs
            gradients_batch = inputs.grad.detach().numpy()
            gradients.append(gradients_batch)


		# Return the gradients
		return np.array(gradients)

	##### Get wrapped model's image shape #####
	def get_image_shape(self):
		input_shape = RESNET_LAYERS_TENSORS['layer1']
		x = input_shape[1]
		y = input_shape[2]
		c = input_shape[0]
		return [x, y, c]

	# Util to get the layer tensors
	def _get_layer_tensors(self):
	  self.layer_tensors = {}
	  self.layers = self.model.layers
	  if self.model_name == 'ResNet50_V2':
	    self.layers = RESNET_LAYERS
	    self.layer_tensors = RESNET_LAYERS_TENSORS


class KerasModelWrapper():
	#----TO DO----
	##### Init #####
	def __init__(self, model_path, labels_path, batch_size):

		self.model_name = None				# Model name
		self.layers = []					# Layer names
		self.layer_tensors = None			# Tensors

		self.simulated_layer_model = {}		# Simulated "layer" model
		self.simulated_logits_model = {}	# Simulated "logits" model

		# Batching
		self.batch_size = batch_size

		# Load model
		self.model = tf.keras.models.load_model(model_path)
		# Fetch tensors
		self._get_layer_tensors()
		# Load labels
		self.labels = tf.io.gfile.GFile(labels_path).read().splitlines()

	##### Get the class label from its id #####
	def id_to_label(self, idx):
		return self.labels[idx]

	##### Get the class id from its label #####
	def label_to_id(self, label):
		return self.labels.index(label)

	##### Get the prediction(s) given one or more input(s) #####
	def get_predictions(self, imgs):

		# Feed the model with the inputs
		inputs = tf.cast(imgs, tf.float32)
		predictions = self.model(inputs)

		# Return the predictions
		return predictions



	##### Get the feature maps given one or more input(s) #####
	def get_feature_maps(self, imgs, layer_name):

		# Simulate a model with the selected layer as the last (lazy)
		if layer_name not in self.simulated_layer_model:
			self.simulated_layer_model[layer_name] = tf.keras.models.Model(
				inputs = [self.model.inputs],
				outputs = [self.layer_tensors[layer_name]]
			)

		# Compute the fmaps
		feature_maps = np.array([])
		for i in range(len(imgs)):
			q = i%self.batch_size
			if q == self.batch_size-1 or i == len(imgs)-1:
				inputs = tf.cast(imgs[i-q : min(i+1, len(imgs))], tf.float32)
				output = self.simulated_layer_model[layer_name](inputs)
				if len(feature_maps) == 0:
					feature_maps = output
				else:
					feature_maps = np.concatenate((feature_maps, output))

		# Return the fmaps
		return feature_maps

	##### Get the logits given a layer and one or more input(s) #####
	def get_logits(self, feature_maps, layer_name):
		#----TO DO----
		# Simulate a model with the logits (lazy)
		if layer_name not in self.simulated_logits_model:
			self.simulated_logits_model[layer_name] = tf.keras.Model(
				inputs = self.layer_tensors[layer_name],
				outputs = self.model.outputs
			)
			self.simulated_logits_model[layer_name].layers[-1].activation = None

		# Feed the model with the inputs
		logits = self.simulated_logits_model[layer_name](feature_maps)

		# Return the logits
		return logits

	##### Get the gradients given a layer and one or more input(s) #####
	def get_gradient_of_score(self, feature_maps, layer_name, target_class_index):
		#----TO DO----
		# Simulate a model with the logits (lazy)
		if layer_name not in self.simulated_logits_model:
			self.simulated_logits_model[layer_name] = tf.keras.Model(
				inputs = self.layer_tensors[layer_name],
				outputs = self.model.outputs
			)
			self.simulated_logits_model[layer_name].layers[-1].activation = None

		# Executing the gradients computation (batching)
		gradients = np.array([])
		for i in range(len(feature_maps)):
			q = i%self.batch_size
			if q == self.batch_size-1 or i == len(feature_maps)-1:
				inputs = tf.cast(feature_maps[i-q : min(i+1, len(feature_maps))], tf.float32)
				# Real batched computation
				with tf.GradientTape() as tape:
					tape.watch(inputs)
					logits = self.simulated_logits_model[layer_name](inputs)
					logit = logits[..., target_class_index]
				output = tape.gradient(logit, inputs)
				# Concatenating the batches' outputs
				if len(gradients) == 0:
					gradients = output
				else:
					gradients = np.concatenate((gradients, output))

		# Return the gradients
		return gradients

	##### Get wrapped model's image shape #####
	def get_image_shape(self):
		input_shape = self.model.input_shape[1:]
		x = input_shape[0]
		y = input_shape[1]
		c = input_shape[2]
		return [x, y, c]

	# Util to get the layer tensors
	def _get_layer_tensors(self):
		self.layer_tensors = {}
		self.layers = self.model.layers
		self.model_name = self.model.name
		for layer in self.layers:
			#print(layer.name)
			#print(self.model_name)
			if 'input' not in layer.name:
				# ResNet50V2
				if self.model_name == 'resnet50v2':
					if "conv4" in layer.name or "conv5" in layer.name:
						if '_out' in layer.name:
							self.layer_tensors[layer.name] = layer.output
					elif layer.name == "post_relu":
						self.layer_tensors[layer.name] = layer.output
				# VGG16
				elif self.model_name == 'vgg16':
					if 'conv' in layer.name and "conv_1" not in layer.name:# and "conv_2" not in layer.name:
						self.layer_tensors[layer.name] = layer.output
				# InceptionV3
				elif self.model_name == 'inception_v3':
					if 'mixed' in layer.name:
						self.layer_tensors[layer.name] = layer.output
				else:
					self.layer_tensors[layer.name] = layer.output

	# Util to reshape the feature maps as needed to feed through the model network
	#def reshape_feature_maps(self, layer_acts):
	#	return np.asarray(layer_acts).squeeze()


#####