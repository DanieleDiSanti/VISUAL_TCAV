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
import os
import numpy as np
from joblib import dump, load
import torchvision
import torch
import torch.nn.functional as F

from TorchModel import Model, TorchModelWrapper, ImageActivationGenerator
from utils import Predictions, Prediction, ConceptLayer, contraharmonic_mean


sys.dont_write_bytecode = True

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGENET_MEAN 	= [0.485, 0.456, 0.406]
IMAGENET_STD 	= [0.229, 0.224, 0.225]

# preprocessing functions
preprocess_resnet_v2 = torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
#preprocess_v3 = tf.keras.applications.inception_v3.preprocess_input
#preprocess_vgg16 = tf.keras.applications.vgg16.preprocess_input


def get_model_by_name(model_name):
	if model_name == 'RESNET50_V2':
		model_graph_path = 'Torch_VisualTCAV/Models/RESNET50/Resnet50_V2.pth'
		model_labels_path = 'Torch_VisualTCAV/Models/RESNET50/ResNet50V2-imagenet-classes.txt'
		return Model(model_name, model_graph_path, model_labels_path)


def get_dtd():
	return torchvision.datasets.DTD(
		root='dtd',
		split='train',
		partition=1,
		transform=None,
		target_transform=None,
		download=True
	)

#####
# VisualTCAV class
#####

class VisualTCAV:

	##### Init #####
	def __init__(
		self,
		model,
		visual_tcav_dir="Torch_VisualTCAV",
		clear_cache=False,
		batch_size=250,
		models_dir=None, cache_dir=None, test_images_dir=None, concept_images_dir=None, random_images_folder=None
	):
		self.tcav_type = 'abstract'
		# Folders and directories
		self.models_dir = os.path.join(visual_tcav_dir, "Models") if not models_dir else models_dir
		self.cache_base_dir = os.path.join(visual_tcav_dir, "cache") if not cache_dir else cache_dir
		self.cache_dir = self.cache_base_dir
		self.test_images_dir = os.path.join(visual_tcav_dir, "test_images") if not test_images_dir else test_images_dir
		self.concept_images_dir = os.path.join(visual_tcav_dir, "concept_images") if not concept_images_dir else concept_images_dir
		self.random_images_folder = "concept_images/random" if not random_images_folder else random_images_folder

		os.makedirs(self.models_dir, exist_ok=True)
		os.makedirs(self.cache_base_dir, exist_ok=True)
		os.makedirs(self.test_images_dir, exist_ok=True)
		os.makedirs(self.concept_images_dir, exist_ok=True)

		self.batch_size = batch_size
		self.predictions = []

		# Model
		self.model = None
		if model:
			self._bind_model(model)

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
	def set_concepts(self, concept_names):
		self.concepts = []
		for concept_name in concept_names:
			if concept_name not in self.concepts:
				self.concepts.append(concept_name)

	# Set a list of layers
	def set_layers(self, layer_names):
		self.layers = []
		for layer_name in layer_names:
			if layer_name not in self.layers:
				self.layers.append(layer_name)

	##### Predict #####
	def predict(self, no_sort=False):

		# Checks
		if not self.tcav_type == 'local':
			raise Exception("Please use a local explainer")
		if not self.model:
			raise Exception("Please instantiate a Model first")

		# Predict with the provided model wrapper
		self.predictions = self.model.model_wrapper.get_predictions(self.resized_imgs)

		# Sort & add class names
		self.predictions = np.array([
			self._sort_target_classes(
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
	def _bind_model(self, model):

		# Folders and directories
		model.graph_path_dir = os.path.join(self.models_dir, model.model_name, model.graph_path_filename)
		model.label_path_dir = os.path.join(self.models_dir, model.model_name, model.label_path_filename)

		# Wrapper function
		model.model_wrapper = TorchModelWrapper(
			model.graph_path_dir,
			model.label_path_dir,
			self.batch_size,
			model.model_name
		)

		# Activate the model
		model.activation_generator = ImageActivationGenerator(
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
	def _sort_target_classes(self, predictions, id_to_label, no_sort=False):

		# Reshape
		indexed_arr = list(enumerate(predictions))
		sorted_arr = indexed_arr if no_sort else sorted(indexed_arr, key=lambda x: x[1], reverse=True)
		return [
			Prediction(
				class_index=sorted_element[0],
				class_name=id_to_label(sorted_element[0]),
				confidence=float(sorted_element[1]),
			) for i, sorted_element in enumerate(sorted_arr) if i < 10
		]

	'''
	feature_maps: Tensor [C,H,W]
	'''
	def _compute_integrated_gradients(self, feature_maps, layer_name, class_index):
		feature_maps = feature_maps.detach().cpu()
		# Generazione degli alphas per l'interpolazione
		alphas = torch.linspace(start=0.0, end=1.0, steps=self.m_steps)  # Alphas per interpolazioni

		# Immagine di baseline con shape uguale a feature_maps
		baseline = torch.zeros_like(feature_maps)  # Baseline di zeri con la stessa forma di fmap

		# Interpolazione delle immagini
		interpolated_images = self._interpolate_images(feature_maps, baseline, alphas)

		# Generazione dei gradienti
		grads = self.model.model_wrapper.get_gradient_of_score(interpolated_images, layer_name, class_index)

		# Calcolo della media dei gradienti per ottenere Integrated Gradients
		avg_grads = torch.mean((grads[:-1] + grads[1:]) / 2.0, dim=0)

		return avg_grads

	# Utils function to interpolate the feature maps
	'''
	feature_maps: Tensor[C,H,W] or Tensor[Batch,C,H,W]
	output: Tensor[Steps,C,H,W] or Tensors[Batch,Steps,C,H,W]
	'''
	def _interpolate_images(self, feature_maps, baseline, alphas):
		if len(feature_maps.shape) == 3:
			alphas_x = alphas.view(-1, 1, 1, 1)
			baseline_x = baseline.unsqueeze(0)
			input_x = feature_maps.unsqueeze(0)

			delta = input_x - baseline_x
			interpolated_images = baseline_x + alphas_x * delta  # (steps, C, H, W)

			return interpolated_images

		elif len(feature_maps.shape) == 4:
			alphas_x = alphas.view(-1, 1, 1, 1, 1)
			baseline_x = baseline.unsqueeze(0)
			input_x = feature_maps.unsqueeze(0)

			delta = input_x - baseline_x
			interpolated_images = baseline_x + alphas_x * delta  # (steps, images, C, H, W)

			return interpolated_images.interpolated_images.permute(1, 0, 2, 3, 4)


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
				dump(random_acts, cache_random_acts_path, compress=3)
		# Return
		return random_acts

	# Compute pooled random
	def _compute_random(self, layer_name):
		feature_maps_for_concept = self.model.activation_generator.get_feature_maps_for_concept(
			'random',
			layer_name,
		)
		return feature_maps_for_concept.detach().cpu()

	# Function to compute the CAV given a concept & a layer
	def _compute_cavs(self, cache, concept_name, layer_name, random_acts=None):

		if random_acts is None:
			random_acts = self._compute_random_activations(cache, layer_name)

		# Define cache path
		cache_path = os.path.join(self.cache_dir, f'cav_{concept_name}_{self.model.max_examples}_{self.random_images_folder}_{layer_name}.joblib')

		# Check if cache exists
		if cache and os.path.isfile(cache_path):
			concept_layer = load(cache_path)
		else:
			# Activations (concept/layer)
			concept_acts = self.model.activation_generator.get_feature_maps_for_concept(concept_name, layer_name).detach().cpu()

			# Pooling the concept and random activations along spatial dimensions
			pooled_concept = concept_acts.mean(dim=(1, 2))
			pooled_random = random_acts.mean(dim=(1, 2))

			# Initialize ConceptLayer
			concept_layer = ConceptLayer()

			# Calculate centroids
			concept_layer.cav.centroid0 = pooled_concept.mean(dim=0)
			concept_layer.cav.centroid1 = pooled_random.mean(dim=0)
			concept_layer.cav.direction = concept_layer.cav.centroid0 - concept_layer.cav.centroid1

			concept_layer.cav.centroid0 = concept_layer.cav.centroid0.detach().cpu()
			concept_layer.cav.centroid1 = concept_layer.cav.centroid1.detach().cpu()
			concept_layer.cav.direction = concept_layer.cav.direction.detach().cpu()

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
