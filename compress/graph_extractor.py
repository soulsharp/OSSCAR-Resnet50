from typing import Optional
from itertools import chain
import copy

import numpy as np
import torch
import torch.fx as fx
from torch import nn

from compress.heuristics import collect_convolution_layers_to_prune
from compress.osscar import get_external_nodes
from compress.osscar_utils import (
    get_coeff_h,
    get_optimal_W,
    get_XtY,
    recompute_H,
    recompute_W,
    reshape_conv_layer_input,
    reshape_filter,
)
from model.resnet import resnet50

def clone_subnet(gm):
    new_graph = torch.fx.Graph()
    env = {}

    for node in gm.graph.nodes:
        def safe_lookup(old_node):
            if old_node in env:
                return env[old_node]
            placeholder = new_graph.placeholder(old_node.name)
            env[old_node] = placeholder
            return placeholder

        new_node = new_graph.node_copy(node, safe_lookup)
        env[node] = new_node

    return torch.fx.GraphModule(gm, new_graph)

def get_initial_prefix_submodule(graph_module, end_node):
    """
    Extracts a prefix subgraph from the start of the FX GraphModule up to (but not including) `end_node`.

    Parameters
    ----------
    graph_module : torch.fx.GraphModule
        The traced FX GraphModule from which to extract the prefix subgraph.
    end_node : str
        The name of the node where the prefix subgraph should stop (exclusive).

    Returns
    -------
    prefix_gm : torch.fx.GraphModule
        A new GraphModule containing only the nodes from the start up to `end_node`.
    value_remap : dict
        A mapping from original nodes in `graph_module` to the corresponding nodes
        in the prefix subgraph. Useful for connecting this prefix to subsequent subgraphs.

    Notes
    -----
    - The resulting GraphModule can be called like a normal module, taking the input tensor
      that corresponds to the placeholder node.
    - The output of the prefix subgraph is the last node before `end_node`.
    """
    assert isinstance(graph_module, fx.GraphModule)
    graph = graph_module.graph
    prefix_nodes = []
    prefix_graph = fx.Graph()
    env = {}

    for node in graph.nodes:
        if node.name == end_node:
            break
        else:
            prefix_nodes.append(node)

    assert len(prefix_nodes) > 0, "Prefix nodes must not be empty"

    for node in prefix_nodes:
        new_node = prefix_graph.node_copy(node, lambda n: env[n] if isinstance(n, fx.Node) else n)
        env[node] = new_node

    last_node = prefix_nodes[-1]
    prefix_graph.output(env[last_node])

    prefix_gm = fx.GraphModule(root=graph_module, graph=prefix_graph)
    return prefix_gm

def get_fx_submodule(gm, start_node, end_node, external_nodes):
    """
    Extracts a subgraph from an FX GraphModule between `start_node` and `end_node`.

    Parameters
    ----------
    graph_module : torch.fx.GraphModule
        The traced FX GraphModule containing the nodes.
    value_remap : dict
        A mapping from previously copied nodes (e.g., from a prefix subgraph) to the
        corresponding new nodes. Must include any nodes that are inputs to this subgraph.
    start_node : str
        The name of the node where the subgraph should start (inclusive).
    end_node : str
        The name of the node where the subgraph should end (exclusive).

    Returns
    -------
    new_gm : torch.fx.GraphModule
        A new GraphModule containing the nodes between `start_node` and `end_node`.
        Automatically adds placeholder nodes for inputs if needed.
    value_remap : dict
        Updated mapping of original nodes to the corresponding nodes in the new subgraph.
        Can be used to chain multiple subgraph extractions together.

    Notes
    -----
    - Placeholder nodes are automatically created for any input nodes that are not in `value_remap`.
    - The output of the subgraph is set to the last node before `end_node`.
    - The resulting GraphModule can be called with the input tensors corresponding to the placeholders.
    """
    assert isinstance(external_nodes, dict), "External nodes must be a dictionary"
    graph = gm.graph
    new_graph = fx.Graph()
    env = {}
    last_old_node = None
    external_deps = list(chain.from_iterable(external_nodes.values()))

    def fetch_arg(n):
        # For non-Node literals
        if not isinstance(n, fx.Node):
            return n

        # ALready exists in env
        if n in env:
            return env[n]

        # Create a placeholder for an external dependency
        if n.name in external_deps:
            ph = new_graph.placeholder(f"external__{n.name}")
            env[n] = ph
            return ph
        
        raise RuntimeError(f"Missing dependency: {n.name}")

    copying = False
    for node in graph.nodes:
        if node.name == start_node:
            copying = True

        if not copying:
            continue

        if node.name == end_node:
            break

        new_node = new_graph.node_copy(node, fetch_arg)
        env[node] = new_node
        last_old_node = new_node

    # Explicitly adds output node
    new_graph.output(env[last_old_node])

    subnet_gm = fx.GraphModule(gm, new_graph)
    return subnet_gm


def get_suffix_submodule(gm, start_node, external_nodes):
    """
    Extracts the subgraph from `start_node` (inclusive) to the final output of the model.

    Parameters
    ----------
    graph_module : fx.GraphModule
        The FX-traced full model.
    start_node : str
        Name of the node where the suffix begins.
    external_nodes : dict
        Dict containing nodes which take inputs that aren't derivable from their own subnet

    Returns
    -------
    suffix_gm : fx.GraphModule
        FX GraphModule for the suffix.
    """
    assert isinstance(external_nodes, dict), "External nodes must be a dictionary"
    graph = gm.graph
    new_graph = fx.Graph()
    env = {}
    last_old_node = None
    external_deps = list(chain.from_iterable(external_nodes.values()))

    def fetch_arg(n):
        # For non-Node literals
        if not isinstance(n, fx.Node):
            return n

        # ALready exists in env
        if n in env:
            return env[n]

        # Create a placeholder for an external dependency
        if n.name in external_deps:
            ph = new_graph.placeholder(f"external__{n.name}")
            env[n] = ph
            return ph
        
        raise RuntimeError(f"Missing dependency: {n.name}")

    copying = False
    for node in graph.nodes:
        if node.name == start_node:
            copying = True

        if not copying:
            continue

        # if node.name == end_node:
        #     break

        new_node = new_graph.node_copy(node, fetch_arg)
        env[node] = new_node
        last_old_node = new_node

    # Explicitly adds output node
    new_graph.output(last_old_node)

    suffix_gm = fx.GraphModule(gm, new_graph)
    return suffix_gm


def get_all_subnets(prune_modules_name, graph_module):
    """
    Build prefix / middle / suffix sub-networks around layers to be pruned.

    Parameters
    ----------
    prune_modules_name : list[str]
        Names of modules in `graph_module` that are candidates for pruning.

    Returns
    -------
    prune_subnets : list[torch.fx.GraphModule]
        A list of GraphModules corresponding to each subnetwork *to be pruned*.
        These subnets may change (weights or architecture) during pruning passes.

    dense_subnets : list[torch.fx.GraphModule]
        A list of GraphModules with exactly the same slices as `prune_subnets`,
        but using the original (dense) architecture and weights. This list remains
        fixed across pruning passes.

    Notes
    -----
    - The sequence of subnets covers: the prefix up to the first prune module,
      the individual prune-module slices, and the suffix after the last prune module.
    - `value_remap` is used internally to maintain graph node identity across slices.
    """
    assert isinstance(
        graph_module, fx.GraphModule
    ), "graph_module must be an instance of fx.GraphModule"
    assert len(prune_modules_name) > 0, "Prune list must not be empty"
    prune_subnets = []
    dense_subnets = []

    gm = graph_module
    external_nodes = get_external_nodes(gm)
    first_node_name = "_".join(prune_modules_name[0].split("."))
    prefix_subnet = get_initial_prefix_submodule(
        graph_module=gm, end_node=first_node_name
    )
    prune_subnets.append(prefix_subnet)
    dense_subnets.append(clone_subnet(prefix_subnet))

    for idx, name in enumerate(prune_modules_name):
        fx_name = "_".join(name.split("."))

        if idx == len(prune_modules_name) - 1:
            subnet= get_suffix_submodule(
                gm=gm, start_node=fx_name, external_nodes=external_nodes
            )
        else:
            end_node = "_".join(prune_modules_name[idx + 1].split("."))
            subnet = get_fx_submodule(
                gm=gm,
                start_node=fx_name,
                end_node=end_node,
                external_nodes=external_nodes
            )

        prune_subnets.append(subnet)
        dense_subnets.append(clone_subnet(subnet))

    return prune_subnets, dense_subnets


def evaluate_loss(submatrix_w_pruned, dense_weights, subgram_xx, subgram_xy):
    """
    Compute the OSSCAR-style loss for a candidate pruned set of channels.

    Parameters
    ----------
    submatrix_w_pruned : torch.Tensor
        Weight matrix corresponding to the pruned channels.
    dense_weights : torch.Tensor
        Original dense weight matrix of the layer.
    subgram_xx : torch.Tensor
        Gram matrix of pruned activations (X^T X for pruned channels).
    subgram_xy : torch.Tensor
        Cross Gram matrix between pruned activations and dense outputs (X^T Y).

    Returns
    -------
    torch.Tensor
        Scalar loss measuring reconstruction error for the pruned channels.
    """
    A = (submatrix_w_pruned.T @ subgram_xx) @ submatrix_w_pruned    
    B = (dense_weights.T @ subgram_xy) @ submatrix_w_pruned
    return torch.trace(A) - 2 * torch.trace(B)


def perform_local_search(
    dense_weights,
    layer,
    p,
    gram_xx,
    gram_xy,
    prune_by_iter: Optional[list] = None,
    sym_diff_per_iter: Optional[list] = None,
    prune_per_iter=2,
):
    """
    Greedy local search to select which input channels to prune in a Conv2d layer.

    Iteratively evaluates the importance of each input channel using Gram matrices
    (X^T X and X^T Y) and prunes the least important channels according to the
    specified schedule.

    Args:
        dense_weights (torch.Tensor): Original dense weight matrix of the layer.
        layer (nn.Conv2d): Convolutional layer to prune.
        p (int): Total number of channels to prune.
        gram_xx (torch.Tensor): Full Gram matrix of layer inputs (X^T X).
        gram_xy (torch.Tensor): Cross Gram matrix between layer inputs and outputs (X^T Y).
        prune_by_iter (list, optional): Custom pruning schedule per iteration.
        sym_diff_per_iter (list, optional): Symmetric difference allowed per iteration.
        prune_per_iter (int, default=2): Number of channels to prune per iteration
            (ignored if `prune_by_iter` is provided).

    Returns:
        keep_mask (torch.BoolTensor): Boolean mask of channels to keep (`True`) or prune (`False`).
        kept_channels (set): Indices of channels retained after pruning.
        removed_channels (set): Indices of channels removed during pruning.
    """
    assert isinstance(layer, nn.Conv2d)
    assert isinstance(p, int) and p > 0

    # Determine pruning schedule
    if prune_by_iter is None:
        prune_list = []
        num_iterations = p // prune_per_iter
        rem = p % prune_per_iter
        if p >= prune_per_iter:
            prune_list.extend([prune_per_iter for _ in range(num_iterations)])
        if rem > 0:
            prune_list.append(rem)
    else:
        assert np.sum(np.array(prune_by_iter)) == p
        prune_list = prune_by_iter

    if sym_diff_per_iter is not None:
        assert len(prune_list) == len(sym_diff_per_iter)
        assert all(p <= t for p, t in zip(prune_list, sym_diff_per_iter))
    else:
        sym_diff_per_iter = prune_list

    kept_channels = set(range(layer.in_channels))
    removed_channels = set()
    total_channels = kept_channels.copy()
    keep_mask = torch.ones(layer.in_channels, dtype=torch.bool)

    # Iterative greedy pruning, where t=p and hence s1 = 0
    for i in range(len(prune_list)):
        num_prune_iter = prune_list[i]
        sym_diff_iter = sym_diff_per_iter[i]
        s1 = (sym_diff_iter - num_prune_iter) // 2
        s2 = (sym_diff_iter + num_prune_iter) // 2

        assert kept_channels.union(removed_channels) == total_channels

        channel_importance_dict = {}

        # Evaluate loss increase if each kept channel were pruned
        for channel in kept_channels:
            temp_keep_mask = keep_mask.clone()
            temp_keep_mask[channel] = False

            subgram_xx = recompute_H(
                prune_mask=temp_keep_mask,
                H=gram_xx,
                kernel_height=layer.kernel_size[0],
                kernel_width=layer.kernel_size[1],
                activation_in_channels=layer.in_channels,
                is_pure_gram=True,
            )
            subgram_xy = recompute_H(
                prune_mask=temp_keep_mask,
                H=gram_xy,
                kernel_height=layer.kernel_size[0],
                kernel_width=layer.kernel_size[1],
                activation_in_channels=layer.in_channels,
                is_pure_gram=False,
            )
            # sub_w_optimal = recompute_W(
            #     prune_mask=temp_keep_mask,
            #     W=w_optimal,
            #     activation_in_channels=layer.in_channels,
            #     kernel_height=layer.kernel_size[0],
            #     kernel_width=layer.kernel_size[1],
            # )
            submatrix_w_pruned = recompute_W(
                prune_mask=temp_keep_mask,
                W=dense_weights,
                activation_in_channels=layer.in_channels,
                kernel_height=layer.kernel_size[0],
                kernel_width=layer.kernel_size[1],
            )

            loss = evaluate_loss(
                submatrix_w_pruned, dense_weights, subgram_xx, subgram_xy
            )
            channel_importance_dict[channel] = loss.item()

        # Sort channels by importance(ascending = least important first)
        sorted_channels = [
            k
            for k, _ in sorted(
                channel_importance_dict.items(), key=lambda item: item[1]
            )
        ]

        # Prune s2 least important channels
        to_prune = sorted_channels[:s2]
        for channel in to_prune:
            keep_mask[channel] = False
            kept_channels.remove(channel)
            removed_channels.add(channel)

        print(
            f"Iteration {i+1}: pruned {len(to_prune)} channels, {len(kept_channels)} remaining."
        )

    return keep_mask, kept_channels, removed_channels


def get_parent_module(model, target_module):
    """Find the direct parent and name of a given submodule.

    Iterates through all modules in `model` to locate the one that directly contains
    `target_module`.

    Args:
        model (nn.Module): Root model to search.
        target_module (nn.Module): Submodule to locate.

    Returns:
        (nn.Module, str): Parent module and the submodule's attribute name.

    Raises:
        ValueError: If the target module isn’t found.
    """
    for _, module in model.named_modules():
        for child_name, child in module.named_children():
            if child is target_module:
                return module, child_name
    raise ValueError("Target module not found")


def replace_module(model, target_module, new_module):
    """Replace a submodule in-place within a model.

    Finds the parent of `target_module` using `get_parent_module` and swaps it with
    `new_module` via `setattr`.

    Args:
        model (nn.Module): Root model.
        target_module (nn.Module): Module to replace.
        new_module (nn.Module): Replacement module.
    """
    parent, name = get_parent_module(model, target_module)
    setattr(parent, name, new_module)


def prune_one_layer(
    dense_subnet, pruned_subnet, dense_input, pruned_input, layer_prune_channels
):
    """
    Prune a single Conv2d layer from a subnet and replace it with a smaller version.

    Uses Gram matrices of the layer inputs and outputs to evaluate channel importance
    and greedily prune the least important input channels. Replaces the original
    layer in `pruned_subnet` with a new Conv2d containing only the kept channels.

    Args:
        dense_subnet (nn.Module): Full, unpruned reference subnet.
        pruned_subnet (nn.Module): Subnet to be pruned and modified.
        dense_input (Tensor): Dense model inputs of shape (num_batches, batch_size, C, H, W).
        pruned_input (Tensor): Pruned model inputs of the same shape.
        layer_prune_channels (int): Number of channels to prune in this layer.

    Returns:
        pruned_subnet (nn.Module): Updated subnet with the pruned layer replaced.
        keep_mask (torch.BoolTensor): Boolean mask indicating which channels were kept.
    """
    assert (
        dense_input.ndim == 5 and pruned_input.ndim == 5
    ), "Inputs must be (num_batches, batch_size, C, H, W)"

    num_batches, batch_size, C, H, W = dense_input.shape
    N = num_batches * batch_size

    # Get conv module to prune
    conv_module = next(
        m for _, m in pruned_subnet.named_modules() if isinstance(m, nn.Conv2d)
    )
    reshaped_conv_wt = reshape_filter(conv_module.weight)

    # Flatten all images into one dimension and reshape for gram_matrices calculation
    dense_input_flat = dense_input.reshape(N, C, H, W)
    pruned_input_flat = pruned_input.reshape(N, C, H, W)
    dense_X = reshape_conv_layer_input(dense_input_flat, conv_module)
    pruned_X = reshape_conv_layer_input(pruned_input_flat, conv_module)

    # Compute gram matrices over all N images
    total_xtx = get_coeff_h(pruned_X) / N
    total_xty = get_XtY(pruned_X, dense_X) / N

    # # Optimal weights
    # w_optimal = get_optimal_W(
    #     gram_xx=total_xtx, gram_xy=total_xty, dense_weights=reshaped_conv_wt
    # )

    keep_mask, kept_channels, removed_channels = perform_local_search(
        dense_weights=reshaped_conv_wt,
        layer=conv_module,
        p=layer_prune_channels,
        gram_xx=total_xtx,
        gram_xy=total_xty,
    )

    # cached_out_pruned = []
    # cached_out_dense = []

    # for batch_idx in range(num_batches):
    #     cached_out_pruned.append(pruned_subnet(pruned_input[batch_idx]))
    #     cached_out_dense.append(dense_subnet(dense_input[batch_idx]))

    # cached_out_pruned = torch.cat(cached_out_pruned, dim=0)
    # cached_out_dense = torch.cat(cached_out_dense, dim=0)

    new_weight = conv_module.weight[:, keep_mask, :, :]
    if conv_module.bias is not None:
        new_bias = conv_module.bias
    else:
        new_bias = None

    kernel_size = conv_module.kernel_size
    stride = conv_module.stride
    padding = conv_module.padding
    dilation = conv_module.dilation

    # Pylance fix
    kernel_size_2 = (kernel_size[0], kernel_size[1])
    stride_2 = (stride[0], stride[1])
    padding_2 = (padding[0], padding[1])
    dilation_2 = (dilation[0], dilation[1])

    # Replacement module
    new_conv_module = nn.Conv2d(
        in_channels=conv_module.in_channels,
        out_channels=conv_module.out_channels - 1,
        kernel_size=kernel_size_2,
        stride=stride_2,
        padding=padding_2,
        dilation=dilation_2,
        groups=conv_module.groups,
        bias=conv_module.bias is not None,
    )

    # Replace weights
    new_conv_module.weight.data = new_weight.clone()
    if new_bias is not None:
        new_conv_module.bias.data = new_bias.clone()

    replace_module(
        model=pruned_subnet, target_module=conv_module, new_module=new_conv_module
    )

    return pruned_subnet, keep_mask


if __name__ == "__main__":
    model = resnet50(pretrained=True)
    input = torch.randn(1, 3, 224, 224)
    weights = model.conv1.weight
    prune_conv_modules, prune_modules_name = collect_convolution_layers_to_prune(
        model=model
    )
    reformatted = []
    for name in prune_modules_name:
        reformatted_name = "_".join(name.split("."))
        reformatted.append(reformatted_name)

    end = reformatted[0]
    gm = fx.symbolic_trace(model)

    prune_subnets, dense_subnets = get_all_subnets(
        graph_module=gm, prune_modules_name=prune_modules_name
    )
    assert len(prune_subnets) == len(dense_subnets)
