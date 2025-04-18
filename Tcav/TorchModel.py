import sys
from torch.utils.data import DataLoader
import numpy as np
from prettytable import PrettyTable
import torch
from joblib import dump, load
from torchvision import datasets, transforms
import os
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from PIL import Image


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

#model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
#weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
#preprocess = weights.transforms()

# Keras preprocessing functions
preprocess_resnet_v2 = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
# preprocess_v3 = tf.keras.applications.inception_v3.preprocess_input
# preprocess_vgg16 = tf.keras.applications.vgg16.preprocess_input
#####
# Model class
#####

RESNET_LAYERS = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'linear']
RESNET18_LAYERS = ['init_block', 'stage1', 'stage2', 'stage3', 'stage4', 'final_pool', 'output']

RESNET_LAYERS_TENSORS = {
    'layer1': (3, 224, 224),
    'layer2': (256, 56, 56),
    'layer3': (512, 28, 28),
    'layer4': (1024, 14, 14),
    'avgpool': (2048, 7, 7),
    'linear': (1, 2048)
}

# index output fmap
VGG_LAYERS_INDEX = {
    'conv1_1': 2,
    'conv1_2': 4,
    'maxpool1': 5,
    'conv2_1': 7,
    'conv2_2': 9,
    'maxpool2': 10,
    'conv3_1': 12,
    'conv3_2': 14,
    'conv3_3': 16,
    'maxpool3': 17,
    'conv4_1': 19,
    'conv4_2': 21,
    'conv4_3': 23,
    'maxpool4': 24,
    'conv5_1': 26,
    'conv5_2': 28,
    'conv5_3': 30,
    'maxpool5': 31
}


def set_batch_size(len_data, size=30):
    while len_data % size != 0:
        size -= 1
    return size


class Model:
    ##### Init #####
    def __init__(self, model_name, graph_path_filename, label_path_filename, preprocessing_function=lambda x: x,
                 binary_classification=False, max_examples=500, model_wrapper=None, activation_generator=None):
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
        self.model_wrapper = model_wrapper
        self.activation_generator = activation_generator
        self.preprocessing_function = preprocessing_function

    ##### Get layer names #####
    def getLayerNames(self):
        return [layer_name for layer_name in self.model_wrapper.layer_tensors.keys()]

    ##### Print model's informations #####
    def info(self):
        # Print a table with information
        table = PrettyTable(title=f"Model: {self.model_name}", field_names=["N. classes", "Layers"], float_format='.2')
        for i, layer_name in enumerate(self.getLayerNames()):
            table.add_row([len(self.model_wrapper.labels) if i == 0 else "", layer_name])
        print(table)


#####
# TorchModelWrapper class
#####

# From Image Input to Feature Maps of a selected Layer
class FeatureMapsModel(nn.Module):
    def __init__(self, model, layer_name, model_name=None):
        super(FeatureMapsModel, self).__init__()
        self.model = model
        self.layer_name = layer_name

        if model_name == 'VGG_16':
            index = VGG_LAYERS_INDEX[layer_name]
            index_feature_extractor = 0
            self.layers = [i for _, i in model.named_children()][index_feature_extractor][0:index]

        else:
            self.layers = nn.Sequential(*list(model.children())[:self._get_layer_index() + 1])

    def _get_layer_index(self):
        layer_names = [name for name, _ in self.model.named_children()]
        if self.layer_name not in layer_names:
            raise ValueError(f"Layer {self.layer_name} not found in the model.")
        return layer_names.index(self.layer_name)

    # Usa DataLoader
    def forward(self, data, detachOutput=False):
        outputs = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()

        if type(data) == DataLoader:
            with torch.no_grad():
                for batch_input, labels in data:
                    batch_input = batch_input.to(device)

                    if detachOutput:
                        batch_output = self.layers(batch_input).detach().cpu()
                    else:
                        batch_output = self.layers(batch_input)

                    outputs.append(batch_output)

            return torch.cat(outputs, dim=0)

        else:
            with torch.no_grad():
                data = data.to(device)
                return self.layers(data)


# From Feature Maps of a selected Layer to Logits
# LayerName must be the same Layer from where the fmap come
class LogitsModel(nn.Module):
    def __init__(self, model, layer_name):
        super(LogitsModel, self).__init__()
        layer_index = RESNET_LAYERS.index(layer_name)
        layer_name = RESNET_LAYERS[layer_index + 1]
        self.model = model
        self.layer_name = layer_name
        self.avg_layer = [i for i in model.modules()][-2]
        self.lin_layer = [i for i in model.modules()][-1]
        # Crea un sottogruppo del modello fino al livello desiderato
        self.conv_layers = nn.Sequential(*list(model.children())[self._get_layer_index():-2])

    def _get_layer_index(self):
        """Trova l'indice del livello specificato nel modello."""
        layer_names = [name for name, _ in self.model.named_children()]
        if self.layer_name not in layer_names:
            raise ValueError(f"Layer {self.layer_name} not found in the model.")
        return layer_names.index(self.layer_name)

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        x = x.to(device)
        self.model.eval()
        if len(self.conv_layers) > 0:
            x = self.conv_layers(x)
        x = self.avg_layer(x).view(x.size(0), -1)  # Flatten before passing to linear layer
        x = self.lin_layer(x)
        return x


class LogitsModel_VGG(nn.Module):
    def __init__(self, model, layer_name):
        super(LogitsModel_VGG, self).__init__()

        self.model = model
        self.layer_name = layer_name
        index = VGG_LAYERS_INDEX[layer_name]
        index_feature_extractor = 0
        index_avg = 1
        index_classifier = 2
        features_extractor = [i for _, i in model.named_children()][index_feature_extractor][index:]
        avg = [i for _, i in model.named_children()][index_avg]
        classifier = [i for _, i in model.named_children()][index_classifier]
        modules = []
        modules.append(features_extractor)
        features_extractor.append(avg)
        modules.append(classifier)
        self.layers = nn.Sequential(*modules)

    def _get_layer_index(self):
        """Trova l'indice del livello specificato nel modello."""
        layer_names = [name for name, _ in self.model.named_children()]
        if self.layer_name not in layer_names:
            raise ValueError(f"Layer {self.layer_name} not found in the model.")
        return layer_names.index(self.layer_name)

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        x = x.to(device)
        self.model.eval()
        x = self.layers[0](x)  # conv layers
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2])  # flatten
        else:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # flatten
        x = self.layers[1](x)  # classifier
        return x


class FeatureMapsModel_Resnet18(nn.Module):

    def __init__(self, model, layer_name, model_name=None):
        super(FeatureMapsModel_Resnet18, self).__init__()
        self.model = model
        self.layer_name = layer_name
        self.layer_index = RESNET18_LAYERS.index(layer_name)

    # Usa DataLoader
    def forward(self, data, detachOutput=False):
        outputs = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()

        if type(data) == DataLoader:
            with torch.no_grad():
                for batch_input, labels in data:
                    x = batch_input.to(device)
                    for i in range(self.layer_index):
                        x = self.model.features._modules[RESNET18_LAYERS[i]](x)

                    batch_output = self.model.features._modules[RESNET18_LAYERS[self.layer_index]](x)

                    if detachOutput:
                        batch_output = batch_output.detach().cpu()

                    outputs.append(batch_output)

            return torch.cat(outputs, dim=0)

        else:
            with torch.no_grad():
                x = data.to(device)
                for i in range(self.layer_index):
                    x = self.model.features._modules[RESNET18_LAYERS[i]](x)

                batch_output = self.model.features._modules[RESNET18_LAYERS[self.layer_index]](x)

                return batch_output


class Resnet18_LogitsModel(nn.Module):
    def __init__(self, model, layer_name):
        super(Resnet18_LogitsModel, self).__init__()
        layer_index = RESNET18_LAYERS.index(layer_name)
        layer_name = RESNET18_LAYERS[layer_index + 1]
        self.model = model
        self.layer_name = layer_name
        self.layer_index = RESNET18_LAYERS.index(layer_name)

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        x = x.to(device)
        self.model.eval()
        for i in range(self.layer_index, len(RESNET18_LAYERS) - 1):
            x = self.model.features._modules[RESNET18_LAYERS[i]](x)

        x = x.squeeze()
        x = self.model.output(x)
        return x



class TorchModelWrapper:
    def __init__(self, model_path, labels_path, batch_size, model_name):
        # Model details
        self.model_name = model_name  # Model name
        self.layers = []  # Layer names
        self.layer_tensors = None  # Tensors

        # Simulated models for specific purposes
        self.simulated_layer_model = {}  # Simulated "layer" model
        self.simulated_logits_model = {}  # Simulated "logits" model

        # Batching
        self.batch_size = batch_size

        # Load model
        self.model = torch.load(model_path, weights_only=False)
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
    def get_predictions(self, data):
        outputs = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        softmax = torch.nn.Softmax(dim=1)  # Softmax lungo l'asse delle classi
        self.model = self.model.to(device)
        self.model.eval()

        if type(data) == DataLoader:
            with torch.no_grad():
                for batch_input, labels in data:
                    batch_input = batch_input.to(device)
                    batch_output = softmax(self.model(batch_input))
                    outputs.append(batch_output.detach().cpu())

            return torch.cat(outputs, dim=0)

        else:
            with torch.no_grad():
                data = data.to(device)
                return softmax(self.model(data)).detach().cpu()

    ##### Get the feature maps given one or more input(s) #####
    def get_feature_maps(self, imgs, layer_name):

        if self.model_name == 'RESNET18':
            f_model = FeatureMapsModel_Resnet18(self.model, layer_name)
        else:
            f_model = FeatureMapsModel(self.model, layer_name, self.model_name)

        '''
        if layer_name not in self.simulated_layer_model:
            self.simulated_layer_model[layer_name] = FeatureMapsModel(self.model, layer_name, self.model_name)
        
        feature_maps = self.simulated_layer_model[layer_name].forward(imgs)
        '''

        feature_maps = f_model.forward(imgs)
        return feature_maps

    ##### Get the logits given a layer and one or more input(s) #####
    def get_logits(self, feature_maps, layer_name):
        if len(feature_maps.shape) == 3:
            feature_maps = feature_maps.unsqueeze(0)

        if self.model_name == 'VGG_16':
            l_model = LogitsModel_VGG(self.model, layer_name)
        if self.model_name == 'RESNET18':
            l_model = Resnet18_LogitsModel(self.model, layer_name)
        else:
            l_model = LogitsModel(self.model, layer_name)

        '''
        # Simulate a model with the logits (lazy)
        if layer_name not in self.simulated_logits_model:
            if self.model_name == 'VGG_16':
                self.simulated_logits_model[layer_name] = LogitsModel_VGG(self.model, layer_name)
            else:
                self.simulated_logits_model[layer_name] = LogitsModel(self.model, layer_name)
        '''

        # Feed the model with the inputs
        logits = l_model.forward(feature_maps)
        #logits = self.simulated_logits_model[layer_name].forward(feature_maps)

        # Return the logits
        return logits

    ##### Get the gradients given a layer and one or more input(s) #####
    def get_gradient_of_score(self, feature_maps, layer_name, target_class_index):
        # Executing the gradients computation (batching)
        gradients = []
        self.batch_size = set_batch_size(len(feature_maps))

        # Process in batches
        for i in range(0, len(feature_maps), self.batch_size):
            inputs = feature_maps[i: i + self.batch_size]
            inputs = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)

            # Forward pass to get logits and compute gradients
            logits = self.get_logits(inputs, layer_name)
            logit = logits[:, target_class_index]  # Select logits for target class

            # Compute gradients
            logit_sum = logit.sum()  # Sum for batch processing
            logit_sum.backward()

            # Retrieve gradients for inputs
            gradients_batch = inputs.grad.detach().cpu()
            gradients.append(gradients_batch)

        # Return the gradients
        return torch.cat(gradients, dim=0)

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
        if self.model_name == 'Resnet50_V2':
            self.layers = RESNET_LAYERS
            self.layer_tensors = RESNET_LAYERS_TENSORS


#####


#####
# ImageActivationGenerator class
#####
class ImageActivationGenerator:
    ##### Init #####
    def __init__(
            self,
            model_wrapper,
            concept_images_dir,
            cache_dir,
            preprocessing_function=None,
            max_examples=500,
    ):
        self.model_wrapper = model_wrapper
        self.concept_images_dir = concept_images_dir
        self.cache_dir = cache_dir
        self.max_examples = max_examples
        self.preprocessing_function = preprocessing_function
        self.batch_size = 30

    def get_feature_maps_for_concept(self, concept, layer, imgs=None, preprocess=True, batch_size=30):
        if imgs is None:
            imgs = self._get_images_for_concept(concept, preprocess=preprocess, batch_size=batch_size)
        feature_maps = self.model_wrapper.get_feature_maps(imgs, layer)
        return feature_maps

    def get_feature_maps_for_layers_and_concepts(self, layer_names, concepts, cache=True, preprocess=True,
                                                 batch_size=30, verbose=False):
        feature_maps = {}
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # For each concept
        for concept in concepts:
            imgs = self._get_images_for_concept(concept, preprocess=preprocess, batch_size=batch_size)
            if concept not in feature_maps:
                feature_maps[concept] = {}

            # For each layer
            for layer_name in layer_names:
                feature_maps_path = os.path.join(self.cache_dir, 'f_maps_{}_{}.joblib'.format(concept,
                                                                                              layer_name)) if self.cache_dir else None

                if feature_maps_path and os.path.exists(feature_maps_path) and cache:
                    if verbose:
                        print(f'Loading from cache: {feature_maps_path}')
                    # Read from cache
                    feature_maps[concept][layer_name] = load(feature_maps_path)

                else:
                    # Compute and write to cache
                    feature_maps[concept][layer_name] = self.get_feature_maps_for_concept(concept, layer_name, imgs)

                    if feature_maps_path and cache:
                        if verbose:
                            print(f'Dumping to cache: {feature_maps_path}')
                        os.makedirs(os.path.dirname(feature_maps_path), exist_ok=True)
                        dump(feature_maps[concept][layer_name], feature_maps_path, compress=2)

        # Return the feature maps
        return feature_maps

    def _get_images_for_concept(self, concept, preprocess=True, batch_size=30, format_DataLoader = True):
        concept_folder = os.path.join(self.concept_images_dir, concept)
        img_folder = self._load_ImageFolder(concept_folder, preprocess=preprocess)

        if format_DataLoader:
            return self._get_DataLoader(img_folder, batch_size)
        else:
            return img_folder

    def _load_ImageFolder(self, images_folder_path, shape=(224, 224), preprocess=True, valid_check=False):
        if self.preprocessing_function is not None and preprocess:
            transform = transforms.Compose([
                transforms.Resize(shape, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.preprocessing_function
            ])

        else:
            transform = transforms.Compose([
                transforms.Resize(shape, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])

        # Carica il dataset utilizzando ImageFolder
        if valid_check:
            x = datasets.ImageFolder(root=images_folder_path, transform=transform, is_valid_file=is_valid_image)
        else:
            x = datasets.ImageFolder(root=images_folder_path, transform=transform)

        return x

    def _get_DataLoader(self, image_folder, batch_size=None, shuffle=False):
        if batch_size is None:
            batch_size = self.batch_size

        n_imgs = len(image_folder)
        if n_imgs % batch_size != 0:
            #print('WARNING: batch_size does not match the number of images!')
            while n_imgs % batch_size != 0:
                batch_size -= 1
            #print(f'New batch_size: {batch_size}')
        return DataLoader(image_folder, batch_size=batch_size, shuffle=shuffle)


def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()  # Verifica se il file è corrotto
        return True
    except (IOError, SyntaxError):
        return False

