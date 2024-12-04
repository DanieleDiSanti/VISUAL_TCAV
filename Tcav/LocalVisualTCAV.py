#####
# Imports
#####

import sys
import PIL.Image
import PIL.ImageFilter
import numpy as np
from matplotlib import pyplot as plt, cm as cm
from matplotlib.gridspec import GridSpec
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import CustomColormap
from VisualTCAV import VisualTCAV

# Do not generate "__pycache__" folder
sys.dont_write_bytecode = True

# Tensorflow
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# preprocessing functions
preprocess_resnet_v2 = torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
# preprocess_v3 = tf.keras.applications.inception_v3.preprocess_input
# preprocess_vgg16 = tf.keras.applications.vgg16.preprocess_input

# Definition
original_colormap = cm.jet
colormap = CustomColormap(
    nodes=[0.0, 0.05] + [i for i in np.linspace(0.1, 1.0, 100)],
    colors=[(0, 0, 0, 1), (0, 0, 0, 1)] + [original_colormap(i) for i in np.linspace(0.15, 1.0, 100)]
)


class LocalVisualTCAV(VisualTCAV):
    # ---TO TEST---
    ##### Init #####
    def __init__(
            self,
            test_image_filename, m_steps=50, n_classes=3, target_class=None,
            *args, **kwargs
    ):

        # Super
        super().__init__(**kwargs)

        self.tcav_type = 'local'

        # Local attributes
        self.test_image_filename = test_image_filename
        self.m_steps = m_steps
        self.target_class = target_class
        if self.target_class is not None:
            self.target_class_index = self.model.model_wrapper.label_to_id(self.target_class)
        elif not self.model.binary_classification:
            self.n_classes = max(np.min([n_classes, len(self.model.model_wrapper.labels), 3]),
                                 1)  # Not implemented more than 3
        else:
            self.n_classes = 2  # add check that it's actually binary

        test_images_path = os.path.join(self.test_images_dir, self.test_image_filename)
        self.resized_imgs_size = self.model.model_wrapper.get_image_shape()[:2]

        self.predictions = []
        self.computations = {}

        img = Image.open(test_images_path).convert('RGB')  # Carica e converte in RGB
        resize_transform = transforms.Resize(self.resized_imgs_size, interpolation=Image.BILINEAR)
        tensor_transform = transforms.ToTensor()

        resized_img = resize_transform(img)
        resized_img = tensor_transform(resized_img)
        self.img = resized_img.permute(1, 2, 0)
        self.resized_img = resized_img.unsqueeze(0)

        ''' Forse per  GlobalTCAV
        self.resized_imgs_size = self.model.model_wrapper.get_image_shape()[:2]
        self.test_images_dir = 'Torch_VisualTCAV/test_images'
        self.resized_imgs = self.model.activation_generator._load_ImageFolder(self.test_images_dir, shape=self.resized_imgs_size)
        self.resized_imgs = self.model.activation_generator._get_DataLoader(self.resized_imgs)
        '''

    def explain(self, cache_cav=True, cache_random=True, cav_only=False):
        # ----TO TEST----
        # Checks
        if not self.model:
            raise Exception("Instantiate a Model first")
        if not self.layers or not self.concepts:
            raise Exception("Please add at least one concept and one layer first")
        if not len(self.predictions):
            raise Exception("Please let the model predict the classes first")

        # Reset the computation variable
        self.computations = {}

        # For each layer
        for layer_name in tqdm(self.layers, desc="Layers", position=0):
            self.computations[layer_name] = {}

            # Random activations
            random_acts = self._compute_random_activations(cache_random, layer_name)

            # Compute the feature maps
            feature_maps = self.model.model_wrapper.get_feature_maps(
                self.model.preprocessing_function(self.resized_img),
                layer_name
            ).detach().cpu()[0]

            # Compute the CAVs
            for concept_name in self.concepts:
                # CAVs
                concept_layer = self._compute_cavs(cache_cav, concept_name, layer_name, random_acts)

                if not cav_only:
                    # Concept map
                    concept_layer.concept_map = F.relu(
                        torch.sum(concept_layer.cav.direction[:, None, None] * feature_maps, dim=0)
                    )

                    # Normalize Concept Map
                    emblem_max, emblem_min = concept_layer.cav.concept_emblem
                    if emblem_max > emblem_min:
                        concept_layer.concept_map = torch.clamp(concept_layer.concept_map, min=emblem_min,
                                                                max=emblem_max)
                        concept_layer.concept_map = (concept_layer.concept_map - emblem_min) / (emblem_max - emblem_min)
                    else:
                        concept_layer.concept_map = concept_layer.concept_map * 0

                # Save the partial computations
                self.computations[layer_name][concept_name] = concept_layer

            if not cav_only:
                # Compute integrated gradients and attributions
                attributions = {}
                for n_class in range(self.n_classes):
                    if not self.model.binary_classification:
                        logits = self.model.model_wrapper.get_logits(feature_maps.unsqueeze(0), layer_name)[0]
                        logits_baseline = self.model.model_wrapper.get_logits(torch.zeros_like(feature_maps).unsqueeze(0), layer_name)[0]

                        ig_expected = F.relu(logits - logits_baseline)

                        ig_expected_max_value = ig_expected.max()
                        ig_expected_norm = ig_expected / ig_expected_max_value if ig_expected_max_value > 0 else ig_expected

                        if self.target_class is not None:
                            ig_expected_class = ig_expected_norm[self.target_class_index]
                        else:
                            ig_expected_class = ig_expected_norm[self.predictions[0][n_class].class_index]

                    # Compute attributions
                    if self.target_class is not None:
                        ig = self._compute_integrated_gradients(feature_maps, layer_name, self.target_class_index)
                    elif self.model.binary_classification:
                        ig = self._compute_integrated_gradients(feature_maps, layer_name,
                                                                self.predictions[0][0].class_index)
                    else:
                        ig = self._compute_integrated_gradients(feature_maps, layer_name,
                                                                self.predictions[0][n_class].class_index)

                    if self.model.binary_classification:
                        binary_attributions = ig * feature_maps
                        virtual_logit_0 = torch.sum(F.relu(binary_attributions))
                        virtual_logit_1 = torch.sum(F.relu(-binary_attributions))
                        max_virtual_logit = max(virtual_logit_0, virtual_logit_1)
                        if max_virtual_logit > 0:
                            virtual_logit_0 /= max_virtual_logit
                            virtual_logit_1 /= max_virtual_logit
                        if n_class == 0:
                            attributions[n_class] = F.relu(binary_attributions)
                            attributions[n_class] = attributions[n_class] * (
                                        virtual_logit_0 / (attributions[n_class].sum() + 1e-10))
                        else:
                            attributions[n_class] = F.relu(-binary_attributions)
                            attributions[n_class] = attributions[n_class] * (
                                        virtual_logit_1 / (attributions[n_class].sum() + 1e-10))
                    else:
                        attributions[n_class] = F.relu(ig * feature_maps)
                        attributions[n_class] = attributions[n_class]*(ig_expected_class/(attributions[n_class].sum()+1e-10))

                # Iterate again on concepts and n_classes
                for concept_name in self.concepts:
                    for n_class in range(self.n_classes):

                        # Mask attributions
                        masked_attributions = attributions[n_class] * self.computations[layer_name][concept_name].concept_map[:, :, None]
                        pooled_masked_attributions = masked_attributions.sum(dim=(0, 1))

                        # Pooled & normalized CAV
                        if feature_maps.min() < 0:
                            pooled_cav_norm = F.relu(
                                self.computations[layer_name][concept_name].cav.direction *
                                torch.where((feature_maps * self.computations[layer_name][concept_name].
                                             concept_map[:, :, None]).sum(dim=(0, 1)) < 0, -1.0, 1.0)
                            )
                        else:
                            pooled_cav_norm = F.relu(self.computations[layer_name][concept_name].cav.direction)

                        max_cav = pooled_cav_norm.max()
                        if max_cav > 0:
                            pooled_cav_norm /= max_cav

                        # Compute and save concept attributions
                        self.computations[layer_name][concept_name].attributions[n_class] = torch.dot(pooled_cav_norm, pooled_masked_attributions)

    ##### Plot heatmaps and information #####
    def plot(self, paper=False):
        # ----TO TEST----
        # Checks
        if not self.model:
            raise Exception("Instantiate a Model first")
        if not self.layers or not self.concepts:
            raise Exception("Please add at least one concept and one layer first")
        if not len(self.predictions):
            raise Exception("Please let the model predict the classes first")
        if not self.computations:
            raise Exception("Please let the model explain first")

        # Escaping
        model_name_esc = self.model.model_name.replace("_", "\_")

        # Iterate over the concepts
        for concept_name in self.concepts:

            # Escaping
            concept_name_esc = concept_name.replace("_", "\_")

            if not paper:
                fig = plt.figure(figsize=(4 + 5 * (len(self.layers) - 1) - 2 * (len(self.layers) - 1), 7))
                gs = GridSpec(3, len(self.layers) * 3, height_ratios=[2, 5, 2])
                fig.suptitle(f"$\mathit{{{model_name_esc}}}$ architecture\n$\mathit{{{concept_name_esc}}}$ concept",
                             fontsize=10 + 1.5 * len(self.layers))
            else:
                fig = plt.figure(figsize=(4 + 5 * (len(self.layers) - 1) - 2 * (len(self.layers) - 1),
                                          6))  # + (0 if self.imgs[0].shape[0] < self.imgs[0].shape[1] else 1)))
                gs = GridSpec(2, len(self.layers) * 3, height_ratios=[1, 3.5])

            # Examples of concepts
            if not paper:
                concept_images = self.model.activation_generator.get_images_for_concept(concept_name, False)
                for i in range(min(len(concept_images), len(self.layers) * 3)):
                    fig.add_subplot(gs[2, i])
                    plt.imshow(concept_images[i])
                    plt.tight_layout()
                    plt.axis('off')

            # Iterate over the layers
            for j, layer_name in enumerate(self.layers):

                # Escaping
                layer_description = "" if len(layer_name) > 11 else "layer"
                layer_name_esc = layer_name.replace("_", "\_")

                # Obtain concept
                concept_layer = self.computations[layer_name][concept_name]

                # Ottieni la mappa del concetto e calcola il massimo valore
                max_value = torch.max(concept_layer.concept_map)

                # Ridimensiona la mappa per ottenere la heatmap
                heatmap = F.interpolate(
                    concept_layer.concept_map.unsqueeze(0).unsqueeze(0),  # Aggiungi dimensioni per batch e canale
                    size=(self.img.shape[0], self.img.shape[1]),
                    mode='bilinear',
                    align_corners=False
                )

                # Rimuovi le dimensioni extra per ottenere la heatmap nel formato desiderato
                heatmap = heatmap.squeeze().numpy()

                # Subplot
                fig.add_subplot(gs[1, j * 3:(j + 1) * 3])
                plt.imshow(self.img)

                # Blurring
                heatmap = np.array(PIL.Image.fromarray(np.uint8(heatmap * 255), 'L').filter(PIL.ImageFilter.GaussianBlur(radius=21))) / 255

                if np.max(heatmap):
                    heatmap = (heatmap / np.max(heatmap)) * max_value

                colormap.imshow(heatmap)
                if not paper:
                    plt.title(f"\n", fontsize=1)
                plt.tight_layout()
                plt.axis('off')

                # Subplot
                fig.add_subplot(gs[0, j * 3:(j + 1) * 3])
                plt.title(f"$\mathit{{{layer_name_esc}}}$ {layer_description}", fontsize=9 + 1.5 * len(self.layers),
                          y=0.95)
                plt.tight_layout()
                rows = []
                for c in range(self.n_classes):
                    attribution = concept_layer.attributions[c]
                    if self.target_class is not None:
                        class_name = self.target_class.replace("-", " ")
                    elif self.model.binary_classification and c == 1:
                        class_name = "Female"  # "Not " + self.predictions[0][0].class_name.replace("-", " ")
                    else:
                        class_name = self.predictions[0][c].class_name.replace("-", " ")
                    if not paper or True:
                        if len(class_name) > 12: class_name = class_name[:12] + "‥"
                    class_name = class_name.replace("_", "\_").replace(" ", "\ ")
                    row = []
                    row.append(f"$\mathit{{{class_name}}}$")
                    if paper:
                        attribution = f"{attribution:.2g}" if (
                                    attribution >= 0.001 or attribution == 0.0) else f"{attribution:.1e}"
                    else:
                        attribution = f"{attribution:.2g}" if attribution >= 0.001 else f"{attribution:.1e}"
                    attribution = attribution.replace("e-0", "e-").replace('-', '{-}')
                    row.append(f"$\mathbf{{{attribution}}}$")
                    rows.append(row)

                cols = [f"$\mathbf{{Class}}$", f"$\mathbf{{Attrib.}}$"]
                table = plt.table(
                    cellText=rows,
                    rowLabels=[f"" for c in range(self.n_classes)],
                    colLabels=cols,
                    rowColours=["silver"] * 10,
                    colColours=["silver"] * 10,
                    cellLoc='center',
                    rowLoc='center',
                    loc='center', edges='BRTL'
                )
                cellDict = table.get_celld()
                for i in range(0, len(rows) + 1):
                    cellDict[(i, 0)].set_width(.625)
                for i in range(0, len(rows) + 1):
                    cellDict[(i, 1)].set_width(.375)
                for i in range(0, len(cols)):
                    cellDict[(0, i)].set_height(.2)
                    for j in range(1, self.n_classes + 1):
                        cellDict[(j, i)].set_height(.2)

                # Set font size
                if paper:
                    table.auto_set_font_size(False)
                    table.set_fontsize(4.5 + 1.75 * len(self.layers))
                else:
                    table.set_fontsize(9 + 1.75 * len(self.layers))

                plt.tight_layout()
                plt.axis('off')

            # Show
            fig.tight_layout()
            plt.show()

    ##### Get CAVs #####
    def getCAVs(self, layer_name, concept_name):

        # Checks
        if not self.model:
            raise Exception("Instantiate a Model first")
        if not self.layers or not self.concepts:
            raise Exception("Please add at least one concept and one layer first")
        if not len(self.predictions):
            raise Exception("Please let the model predict the classes first")
        if not self.computations:
            raise Exception("Please let the model explain first")

        return self.computations[layer_name][concept_name].cav
