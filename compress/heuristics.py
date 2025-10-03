import math

import numpy as np
import torch
from torch import nn

from model.resnet import resnet50
from utils.utils import count_parameters


def compute_layer_importance_heuristic(layers):
    """
    Compute a simple importance score for each convolutional layer
    proportional to its parameter count.

    Parameters
    ----------
    layers : list of nn.Conv2d
        Convolutional layers to evaluate.

    Returns
    -------
    numpy.ndarray
        Normalized importance scores for each layer (sum to 1).
    """
    importance_list = []
    for conv_layer in layers:
        assert isinstance(
            conv_layer, nn.Conv2d
        ), "This method only prunes Conv layers at the moment"
        num_parameters = count_parameters(conv_layer, in_millions=False)
        importance_list.append(num_parameters)

    total = np.sum(importance_list, axis=0)
    importance_list = importance_list / total

    return importance_list


def collect_convolution_layers_to_prune(model: nn.Module):
    """
    Collect convolutional layers from a model that are eligible for pruning.

    By default this picks all modules whose name contains 'conv' but does not
    end with 'conv3' (e.g. skips the final 1x1 conv in ResNet bottlenecks).

    Parameters
    ----------
    model : nn.Module
        Model to scan for convolutional layers.

    Returns
    -------
    tuple
        prune_conv_modules : list of nn.Conv2d
            Convolutional layers selected for pruning.
        prune_conv_modules_name : list of str
            Names of the selected layers (as in model.named_modules()).
    """
    prune_conv_modules = []
    prune_conv_modules_name = []
    for name, module in model.named_modules():
        if "conv" in name and not name.endswith("conv3"):
            prune_conv_modules.append(module)
            prune_conv_modules_name.append(name)

    return prune_conv_modules, prune_conv_modules_name


if __name__ == "__main__":
    model = resnet50(pretrained=False)
    prune_conv_modules, _ = collect_convolution_layers_to_prune(model=model)
    print("Weights_shape: \n", prune_conv_modules[0].weight.shape)

    importance_list = compute_layer_importance_heuristic(prune_conv_modules)
    assert math.isclose(
        np.sum(importance_list), 1.0, abs_tol=0.00001
    ), "importance scores must sum to 1"
    assert len(prune_conv_modules) == len(
        importance_list
    ), "Num modules to be pruned must match the importance list"
    print(importance_list)
