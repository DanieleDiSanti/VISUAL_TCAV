import sys
import os
import numpy as np
import torch
import torchvision
from prettytable import PrettyTable
from matplotlib import pyplot as plt, cm as cm
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F

from Tcav.VisualTCAV import VisualTCAV
from Tcav.utils import Stat

sys.dont_write_bytecode = True


# Tensorflow
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGENET_MEAN 	= [0.485, 0.456, 0.406]
IMAGENET_STD 	= [0.229, 0.224, 0.225]

# Keras preprocessing functions
preprocess_resnet_v2 = torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
#preprocess_v3 = tf.keras.applications.inception_v3.preprocess_input
#preprocess_vgg16 = tf.keras.applications.vgg16.preprocess_input


class GlobalVisualTCAV(VisualTCAV):

	##### Init #####
	def __init__(
		self,
		target_class, test_images_folder, m_steps=50, compute_negative_class=False,
		*args, **kwargs
	):

		# Super
		super().__init__(**kwargs)

		self.tcav_type = 'global'

		# Local attributes
		self.m_steps = m_steps
		self.target_class = target_class
		self.compute_negative_class = compute_negative_class
		self.test_images_folder = test_images_folder
		self.test_image_filename = test_images_folder
		self.class_index = self.model.model_wrapper.label_to_id(target_class)

		#self.test_images_dir = os.path.join(self.test_images_dir, self.test_images_folder)
		self.resized_imgs_size = self.model.model_wrapper.get_image_shape()[:2]

		self.predictions = []
		self.stats = {}

	##### Explain #####
	def explain(self, cache_cav=True, cache_random=True):
		# Checks
		if not self.model:
			raise Exception("Instantiate a Model first")
		if not self.layers or not self.concepts:
			raise Exception("Please add at least one concept and one layer first")

		# Reset the computation variable
		self.stats = {}

		# For each layer
		for layer_name in tqdm(self.layers, desc="Layers", position=0):
			self.stats[layer_name] = {}

			# Random activations
			random_acts = self._compute_random_activations(cache_random, layer_name)

			# Compute the feature_maps for each class
			class_feature_maps = self.computeFeatureMaps(layer_name)

			# For each concept
			cavs = {}
			attribution_list = {}
			for concept_name in self.concepts:
				# CAVs
				concept_layer = self._compute_cavs(cache_cav, concept_name, layer_name, random_acts)

				# Save the partial computations
				cavs[concept_name] = concept_layer
				attribution_list[concept_name] = {}

			# For each image
			for cl, feature_maps in enumerate(tqdm(class_feature_maps, desc="Attributions", position=1)):

				if not self.model.binary_classification:
					# Compute logits
					logits = self.model.model_wrapper.get_logits(torch.unsqueeze(feature_maps, dim=0), layer_name).detach().cpu()[0]
					logits_baseline = self.model.model_wrapper.get_logits(torch.unsqueeze(torch.zeros_like(feature_maps), dim=0), layer_name).detach().cpu()[0]

					ig_expected = F.relu(logits - logits_baseline)

					ig_expected_max_value = torch.max(ig_expected)
					if ig_expected_max_value > 0:
						ig_expected_norm = ig_expected / ig_expected_max_value
					else:
						ig_expected_norm = ig_expected

					ig_expected_class = ig_expected_norm[self.class_index]

				# Compute attributions
				ig = self._compute_integrated_gradients(feature_maps, layer_name, self.class_index)
				if self.model.binary_classification:
					# Binary classification case
					binary_attributions = ig * feature_maps
					virtual_logit_0 = torch.sum(F.relu(binary_attributions))
					virtual_logit_1 = torch.sum(F.relu(-binary_attributions))
					max_virtual_logit = max(virtual_logit_0, virtual_logit_1)
					if max_virtual_logit > 0:
						virtual_logit_0 /= max_virtual_logit
						virtual_logit_1 /= max_virtual_logit
					if not self.compute_negative_class:
						attributions = F.relu(binary_attributions)
						attributions = (attributions / (torch.sum(attributions) + 1e-10)) * virtual_logit_0
					else:
						attributions = F.relu(-binary_attributions)
						attributions = (attributions / (torch.sum(attributions) + 1e-10)) * virtual_logit_1
				else:
					attributions = F.relu(ig * feature_maps)
					attributions = (attributions / (torch.sum(attributions) + 1e-10)) * ig_expected_class

				# Again for each concept
				for concept_name in self.concepts:
					# Concept map
					concept_map = F.relu(
						torch.sum(cavs[concept_name].cav.direction[:, None, None] * feature_maps, dim=0)
					)

					# Normalize Concept Map
					if cavs[concept_name].cav.concept_emblem[0] > cavs[concept_name].cav.concept_emblem[1]:
						concept_map = torch.clamp(
							concept_map,
							min=cavs[concept_name].cav.concept_emblem[1],
							max=cavs[concept_name].cav.concept_emblem[0]
						)
						concept_map = (concept_map - cavs[concept_name].cav.concept_emblem[1]) / (
								cavs[concept_name].cav.concept_emblem[0] - cavs[concept_name].cav.concept_emblem[1]
						)
					else:
						concept_map *= 0

					# Mask attributions
					pooled_masked_attributions = torch.sum(
						attributions * concept_map[None, :, :], dim=(1, 2)
					)

					# Pooled & normalized CAV
					if torch.min(feature_maps) < 0:
						pooled_cav_norm = F.relu(
							cavs[concept_name].cav.direction *
							torch.where(
								torch.sum(feature_maps * concept_map[None, :, :], dim=(1, 2)) < 0, -1.0, 1.0)
						)
					else:
						pooled_cav_norm = F.relu(cavs[concept_name].cav.direction)

					max_cav = torch.max(pooled_cav_norm)
					if max_cav > 0:
						pooled_cav_norm /= max_cav

					# Compute and save concept attributions
					attribution_list[concept_name][cl] = torch.tensordot(pooled_cav_norm, pooled_masked_attributions, dims=1)

			# Again for each concept
			for concept_name in self.concepts:
				# Compute stats
				self.stats[layer_name][concept_name] = Stat(list(attribution_list[concept_name].values()))

			# Clear memory
			del cavs
			del attribution_list

	##### Plot graphs and information #####
	def plot(self, palette='RdYlBu_r'):
		#----TO DO----
		# Checks
		if not self.model:
			raise Exception("Instantiate a Model first")
		if not self.layers or not self.concepts:
			raise Exception("Please add at least one concept and one layer first")
		if not self.stats:
			raise Exception("Please let the model explain first")

		# Colors
		colors = sns.color_palette(palette=palette, n_colors=len(self.layers))

		# Escaping
		model_name_esc = self.model.model_name.replace("_", "\_").replace("-", "{-}").replace(" ", "\\text{ }")
		target_class_esc = self.target_class.replace("_", "\_").replace("-", "{-}").replace(" ", "\\text{ }")
		if self.model.binary_classification and self.compute_negative_class==True:
			target_class_esc = 'Female'#"Not " + target_class_esc
		concept_names = [concept.replace("_", "\_").replace("-", "{-}").replace(" ", "\\text{ }") for concept in self.concepts]

		# Figure
		fig = plt.figure(figsize=(5 + 1*(len(self.concepts)-1), 4))
		gs = GridSpec(1, 1, height_ratios=[1])
		fig.suptitle(f"$\mathit{{{model_name_esc}}}$ architecture\n$\mathit{{{target_class_esc}}}$ target class", fontsize=12)
		# Subplot
		fig.add_subplot(gs[0])

		# Axes
		x = np.arange(len(self.concepts))-0.5

		# Iterate over the concepts
		for i, layer_name in enumerate(self.layers):

			# Indexing
			color = i / (len(self.layers)-1) if len(self.layers) > 1 else 0.5
			width = 0.1
			pos_x = 0.5 + (i-len(self.layers)/2)*width + width/2

			# Escaping
			layer_name_esc = layer_name.replace("_", "\_").replace("-", "{-}").replace(" ", "\\text{ }")

			# Bar
			plt.bar(
				x+pos_x,
				[(self.stats[layer_name][concept_name].begin + self.stats[layer_name][concept_name].end)/2 for concept_name in self.concepts],
				yerr=[max(0, (self.stats[layer_name][concept_name].end - self.stats[layer_name][concept_name].begin)/2) for concept_name in self.concepts],
				width=width,
				label=f'$\mathit{{{layer_name_esc}}}$',
				zorder = 2,
				capsize = 3.5,
				#color=cmap(((color)/8)*6 + 1/8),
				color=colors[i]
			)

		# Show
		#plt.xlabel('Concept')
		plt.ylabel('Attribution (2σ error)')
		plt.xticks(np.arange(len(self.concepts)), [f'$\mathit{{{concept}}}$' for concept in concept_names])
		plt.grid(linewidth=0.3, zorder=1)
		plt.legend(bbox_to_anchor=(1.025, 1.0), loc='upper left', borderaxespad=0.0)
		#plt.legend()
		fig.tight_layout()
		plt.ylim(bottom=0, top=max(0.1, plt.ylim()[1]))
		plt.xlim(left=-0.5, right=0.5 + len(self.concepts)-1)
		plt.show()

	##### Print stats and information #####
	def statsInfo(self):

		# Checks
		if not self.model:
			raise Exception("Instantiate a Model first")
		if not self.layers or not self.concepts:
			raise Exception("Please add at least one concept and one layer first")
		if not self.stats:
			raise Exception("Please let the model explain first")

		# Print a table with information
		table = PrettyTable(title=f"Model: {self.model.model_name}; Class: {self.target_class}; Examples: {self.test_images_folder}", field_names=["Concept", "Layer", "Attrib. mean", "Attrib. 95.45% CI"], float_format='.2')
		for i, concept_name in enumerate(self.concepts):
			for j, layer_name in enumerate(self.layers):
				table.add_row([
						concept_name if j == 0 else "", layer_name,
						f"{self.stats[layer_name][concept_name].mean:.3g} +- {self.stats[layer_name][concept_name].std:.3g}",
						[f"{self.stats[layer_name][concept_name].begin:.3g}", f"{self.stats[layer_name][concept_name].end:.3g}"],
					],
					#divider=True if j == len(self.layers)-1 else False,
				)
		print(table)

	##### Function used to compute the FEATURE MAPS #####
	def computeFeatureMaps(self, layer_name):

		# Checks
		if not self.model:
			raise Exception("Instantiate a Model first")
		if not layer_name:
			raise Exception("Please provide the function with one layer")

		# Compute the feature maps for each class
		self.model.activation_generator.concept_images_dir = self.test_images_dir
		class_feature_maps = self.model.activation_generator.get_feature_maps_for_concept(self.test_images_folder, layer_name)
		self.model.activation_generator.concept_images_dir = self.concept_images_dir

		return class_feature_maps
