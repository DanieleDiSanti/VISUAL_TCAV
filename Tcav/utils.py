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
import numpy as np
from prettytable import PrettyTable
from matplotlib import pyplot as plt, cm as cm
import torch

sys.dont_write_bytecode = True


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
	x = torch.square(arr)
	numerator = torch.sum(x, axis=axis)
	denominator = torch.sum(arr, axis=axis)
	return torch.divide(numerator, (torch.add(denominator, 1e-10)))


#####
# ConceptLayer class
#####
class ConceptLayer:

	##### Init #####
	def __init__(self):

		# Attributes
		self.attributions = {}
		self.concept_map = None

		# CAV
		self.cav = Cav()


#####
# Cav class
##### ##

class Cav:

	##### Init #####
	def __init__(self, direction=None, centroid0=None, centroid1=None, concept_emblem=None):

		# Attributes
		self.direction = direction
		self.centroid0 = centroid0
		self.centroid1 = centroid1
		self.concept_emblem = concept_emblem


#####
# Prediction class
#####

class Prediction:

	##### Init #####
	def __init__(self, class_name=None, class_index=None, confidence=None):

		# Attributes
		self.class_name = class_name
		self.class_index = class_index
		self.confidence = confidence


#####
# Predictions class
#####

class Predictions:

	##### Init #####
	def __init__(self, predictions, test_image_filename, model_name):

		# Attributes
		self.predictions = predictions
		self.test_image_filename = test_image_filename
		self.model_name = model_name

	##### Plot a table with the predictions information #####
	def info(self, num_of_classes = 3):
		# Print a table with information
		table = PrettyTable(title=f"Model: {self.model_name}", field_names=["Image", "Class name", "Confidence"], float_format='.2')
		for i in range(min(num_of_classes, len(self.predictions[0]))):
			table.add_row([
				self.test_image_filename if i == 0 else "",
				self.predictions[0][i].class_name,
				f"{self.predictions[0][i].confidence:.2g}"
			])
		print(table)


#####
# Stat class
#####

class Stat:

	##### Init #####
	def __init__(self, attributions):
		# Attributes
		self.attributions = attributions

		# Simple
		self.mean = torch.mean(torch.tensor(self.attributions, dtype=torch.float32))
		self.std = torch.tensor(np.std(self.attributions, ddof=1), dtype=torch.float32)

		# Compute confidence interval
		self.n = len(self.attributions)
		self.std_err = self.std / torch.sqrt(torch.tensor(self.n, dtype=torch.float32))
		self.begin = torch.relu(self.mean - self.std_err * 2)
		self.end = self.mean + self.std_err * 2


#####
# CustomColormap class
#####

class CustomColormap:

	##### Init #####
	def __init__(self, nodes=None, colors=None, min=0, max=1, alpha=0.6):
		# Error handling
		if type(nodes) == type(colors):
			if hasattr(nodes, "__len__") and hasattr(nodes, "__len__"):
				if len(nodes) != len(colors):
					raise ValueError('Arrays of different lengths')
			elif nodes is not None or colors is not None:
				raise ValueError('Type not supported')
		else:
			raise ValueError('Attributes of different types')
		if min >= max:
			raise ValueError
		# Set attributes
		self.nodes = nodes
		self.colors = colors
		self.min = min
		self.max = max
		self.alpha = alpha

	##### Get LinearSegmentedColormap #####
	def getLinearSegmentedColormap(self):
		from matplotlib.colors import LinearSegmentedColormap
		return LinearSegmentedColormap.from_list("custom", list(zip(self.nodes, self.colors)))

	##### Plot imshow #####
	def imshow(self, heatmap):
		plt.imshow(
			heatmap,
			cmap=self.getLinearSegmentedColormap(),
			alpha=self.getAlpha(),
			vmin=self.getMin(),
			vmax=self.getMax()
		)
		plt.clim(self.getMin(), self.getMax())
		#plt.colorbar(shrink=0.8)

	# Getters
	def getMin(self):
		return self.min
	def getMax(self):
		return self.max
	def getAlpha(self):
		return self.alpha


def show_activation_maps(img, layers, use_mean_vec=True, cmap='inferno'):
	n_layers = len(layers)
	fig, axes = plt.subplots(n_layers + 1, 2, figsize=(10, 15), gridspec_kw={'width_ratios': [10, 1]})

	# Original Image
	title = f'Image\nShape {np.array(img.detach().cpu()).shape}'
	axes[0, 0].set_title(title)
	cax = axes[0, 0].imshow(img.permute(1, 2, 0), cmap=cmap)
	fig.colorbar(cax, ax=axes[0, 0], orientation='vertical')
	cax = axes[0, 1].imshow(np.ones((10, 1)), cmap=cmap)
	fig.colorbar(cax, ax=axes[0, 1], orientation='vertical')
	axes[0, 1].xaxis.set_visible(False)

	# Feature Maps and Feature Vectors
	for layer in range(0, n_layers):
		layer_model = layers[layer]
		row = layer + 1

		fmaps = layer_model.forward(img.unsqueeze(0))[0]
		mean_values = fmaps.mean(axis=(1, 2))
		max_index = torch.argmax(mean_values)

		# Mostra la feature map del primo canale
		title = f'Layer {layer + 1} - F_map N° {max_index}\n Shape {fmaps.squeeze().numpy().shape}'
		axes[row, 0].set_title(title)
		cax = axes[row, 0].imshow(fmaps[max_index], cmap=cmap)
		fig.colorbar(cax, ax=axes[row, 0], orientation='vertical', label=f"Activation Value")

		# Calcola i valori medi o massimi per ogni canale
		if use_mean_vec:
			label = 'Mean'
			mean_values = fmaps.mean(axis=(1, 2))
		else:
			label = 'Max'
			mean_values = fmaps.max(axis=(1, 2))

		# prendi solo le prime 10 fmaps per semplicità
		mean_values_matrix = mean_values[max_index - 4:max_index + 5].reshape(-1, 1)

		ax = axes[row, 1]
		# Visualizza la matrice di calore
		cax = axes[row, 1].imshow(mean_values_matrix, cmap=cmap, aspect='auto')
		ax.xaxis.set_visible(False)
		ax.set_ylabel("F_map")
		ax.set_title("Activation Vector")
		fig.colorbar(cax, ax=axes[row, 1], orientation='vertical', label=f"{label} Activation Value")

	# Imposta layout per evitare sovrapposizioni
	plt.tight_layout()
	plt.show()

# Definition
original_colormap = cm.jet
colormap = CustomColormap(
	nodes=[0.0, 0.05] + [i for i in np.linspace(0.1, 1.0, 100)],
	colors=[(0, 0, 0, 1), (0, 0, 0, 1)] + [original_colormap(i) for i in np.linspace(0.15, 1.0, 100)]
)
