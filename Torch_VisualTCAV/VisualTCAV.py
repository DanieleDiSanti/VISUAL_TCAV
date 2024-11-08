#####
#	VisualTCAV
#
#	All rights reserved.
#
#	Main classes
#####


#####
# Imports
#####

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

# Utils
def cosine_similarity(vec1, vec2):
	dot_product = np.dot(vec1, vec2)
	norm_vec1 = np.linalg.norm(vec1)
	norm_vec2 = np.linalg.norm(vec2)
	return dot_product / (norm_vec1 * norm_vec2)

def nth_highest_index(arr, n):
    indexed_arr = list(enumerate(arr))
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)
    return sorted_arr[n-1][0]

def contraharmonic_mean(arr, axis=(0, 1)):
	#-----TO TEST--------
	numerator = torch.sum(torch.square(arr), axis=axis)
	denominator = torch.sum(arr, axis=axis)
	return torch.divide(numerator, (torch.add(denominator, tf.keras.backend.epsilon())))

#####
# VisualTCAV class
#####

class VisualTCAV():

	##### Init #####
	def __init__(
		self,
		model,
		visual_tcav_dir="VisualTCAV",
		clear_cache=False,
		batch_size=250,
		models_dir=None, cache_dir=None, test_images_dir=None, concept_images_dir=None, random_images_folder=None
	):
		# Folders and directories
		self.models_dir = os.path.join(visual_tcav_dir, "models") if not models_dir else models_dir
		self.cache_base_dir = os.path.join(visual_tcav_dir, "cache") if not cache_dir else cache_dir
		self.cache_dir = self.cache_base_dir
		self.test_images_dir = os.path.join(visual_tcav_dir, "test_images") if not test_images_dir else test_images_dir
		self.concept_images_dir = os.path.join(visual_tcav_dir, "concept_images") if not concept_images_dir else concept_images_dir
		self.random_images_folder = "random" if not random_images_folder else random_images_folder

		os.makedirs(self.models_dir, exist_ok=True)
		os.makedirs(self.cache_base_dir, exist_ok=True)
		os.makedirs(self.test_images_dir, exist_ok=True)
		os.makedirs(self.concept_images_dir, exist_ok=True)

		self.batch_size = batch_size

		# Model
		self.model = None
		if model:
			self._bindModel(model)

		if clear_cache:
			for file in os.listdir(self.cache_dir):
				os.remove(os.path.join(self.cache_dir, file))

		# Concepts/Layers attributes
		self.concepts = []
		self.layers = []

		# Computations
		self.computations = {}
		self.random_acts = {}

	# Set a list of concepts
	def setConcepts(self, concept_names):
		self.concepts = []
		for concept_name in concept_names:
			if concept_name not in self.concepts:
				self.concepts.append(concept_name)

	# Set a list of layers
	def setLayers(self, layer_names):
		self.layers = []
		for layer_name in layer_names:
			if layer_name not in self.layers:
				self.layers.append(layer_name)

	##### Predict #####
	def predict(self, no_sort=False):

		# Checks
		if not isinstance(self, LocalVisualTCAV):
			raise Exception("Please use a local explainer")
		if not self.model:
			raise Exception("Please instantiate a Model first")

		# Predict with the provided model wrapper
		self.predictions = self.model.model_wrapper.get_predictions(
			self.model.preprocessing_function(
				self.resized_imgs
			)
		)

		# Sort & add class names
		self.predictions = np.array([
			self._sortTargetClasses(
				prediction,
				self.model.model_wrapper.id_to_label,
				no_sort
			) for prediction in self.predictions
		])

		# Return the classes
		return Predictions(self.predictions, self.test_image_filename, self.model.model_name)


	#####
	# Private methods
	#####

	# Bind a model
	def _bindModel(self, model):

		# Folders and directories
		model.graph_path_dir = os.path.join(self.models_dir, model.model_name, model.graph_path_filename)
		model.label_path_dir = os.path.join(self.models_dir, model.model_name, model.label_path_filename)

		# Wrapper function
		model.model_wrapper = model.model_wrapper(model.graph_path_dir, model.label_path_dir, self.batch_size)

		# Activate the model
		model.activation_generator = model.activation_generator(
			model_wrapper=model.model_wrapper,
			concept_images_dir=self.concept_images_dir,
			cache_dir=self.cache_dir,
			preprocessing_function=model.preprocessing_function,
			max_examples=model.max_examples,
		)

		# Model's cache dir
		self.cache_dir = os.path.join(self.cache_base_dir, model.model_name)
		os.makedirs(self.cache_dir, exist_ok=True)

		# Store the model
		self.model = model

	# Reshape a list of predictions
	def _sortTargetClasses(self, predictions, id_to_label, no_sort=False):

		# Reshape
		indexed_arr = list(enumerate(predictions))
		sorted_arr = indexed_arr if no_sort else sorted(indexed_arr, key=lambda x: x[1], reverse=True)
		return [
			Prediction(
				class_index=sorted_element[0],
				class_name=id_to_label(sorted_element[0]),
				confidence=sorted_element[1],
			) for i, sorted_element in enumerate(sorted_arr) if i < 10
		]

	def _compute_integrated_gradients(self, feature_maps, layer_name, class_index):
		#---TO TEST---
		# Generazione degli alphas per l'interpolazione
		alphas = torch.linspace(start=0.0, end=1.0, steps=self.m_steps + 1)  # m_steps intervalli per la Riemann sum

		# Immagine di baseline con shape uguale a feature_maps
		baseline = torch.zeros_like(feature_maps)

		# Interpolazione delle immagini
		interpolated_images = self._interpolate_images(feature_maps, baseline, alphas)

		# Generazione dei gradienti
		grads = self.model.model_wrapper.get_gradient_of_score(interpolated_images, layer_name, class_index)

		# Calcolo della media dei gradienti per ottenere Integrated Gradients
		grads = torch.stack(grads)  # Assicuriamoci che grads sia un tensore di PyTorch
		avg_grads = torch.mean((grads[:-1] + grads[1:]) / 2.0, dim=0)

    return avg_grads


	# Utils function to interpolate the feature maps
	def _interpolate_images(self, feature_maps, baseline, alphas):
		#----TO TEST----
    # Converting feature_maps to float32
    image = feature_maps.float()

    # Expanding dimensions to match the shape of the tensors
    alphas_x = alphas.view(-1, 1, 1, 1)
    baseline_x = baseline.unsqueeze(0)
    input_x = image.unsqueeze(0)

    # Calculating the delta and interpolated images
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta

    return images

	# Function to compute the negative examples activations for a given layer
	def _compute_random_activations(self, cache, layer_name):

		# Random activations
		cache_random_acts_path = os.path.join(self.cache_dir, 'rnd_acts_' + str(self.model.max_examples) + "_" + self.random_images_folder + '_' + layer_name + '.joblib')
		if cache and os.path.isfile(cache_random_acts_path):
			random_acts = load(cache_random_acts_path)
		else:
			random_acts = self._compute_random(layer_name)

			# If cache is requested
			if cache:
				dump(random_acts, cache_random_acts_path, compress=9)
		# Return
		return random_acts

	# Compute pooled random
	def _compute_random(self, layer_name):
		feature_maps_for_concept = self.model.activation_generator.get_feature_maps_for_concept(
			self.random_images_folder,
			layer_name,
		)
		return feature_maps_for_concept


	# Function to compute the CAV given a concept & a layer
	def _compute_cavs(self, cache, concept_name, layer_name, random_acts):
		#---TO TEST---
		# Define cache path
    cache_path = os.path.join(self.cache_dir, f'cav_{concept_name}_{self.model.max_examples}_{self.random_images_folder}_{layer_name}.joblib')

    # Check if cache exists
    if cache and os.path.isfile(cache_path):
        concept_layer = load(cache_path)
    else:
        # Activations (concept/layer)
        concept_acts = self.model.activation_generator.get_feature_maps_for_concept(concept_name, layer_name)

        # Pooling the concept and random activations along spatial dimensions
        pooled_concept = concept_acts.mean(dim=(1, 2))
        pooled_random = random_acts.mean(dim=(1, 2))

        # Initialize ConceptLayer
        concept_layer = ConceptLayer()

        # Calculate centroids
        concept_layer.cav.centroid0 = pooled_concept.mean(dim=0)
        concept_layer.cav.centroid1 = pooled_random.mean(dim=0)
        concept_layer.cav.direction = concept_layer.cav.centroid0 - concept_layer.cav.centroid1

        # Emblems computation
        emblems = contraharmonic_mean(
            F.relu(
                (concept_layer.cav.direction[None, None, None, :] * concept_acts).sum(dim=3)
            ),
            axis=(1, 2)
        )

        negative_emblems = contraharmonic_mean(
            F.relu(
                (concept_layer.cav.direction[None, None, None, :] * random_acts).sum(dim=3)
            ),
            axis=(1, 2)
        )

        # Convert emblems to float tensors
        concept_layer.cav.concept_emblem = torch.tensor([
            torch.quantile(emblems, 0.5),
            torch.quantile(negative_emblems, 0.5)
        ], dtype=torch.float32)

        print(concept_layer.cav.concept_emblem)

        # Cache the computed concept layer if requested
        if cache:
            dump(concept_layer, cache_path, compress=9)

    # Return the computed concept layer
    return concept_layer
