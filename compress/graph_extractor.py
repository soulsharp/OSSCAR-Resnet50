from typing import Optional
import numpy as np
import torch
import torch.fx as fx
from torch import nn

from compress.heuristics import collect_convolution_layers_to_prune
from compress.osscar_utils import (
    get_XtY,
    get_coeff_h,
    reshape_conv_layer_input,
    reshape_filter,
    get_optimal_W,
)
from model.resnet import resnet50


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
    value_remap = {}

    for node in graph.nodes:
        if node.name == end_node:
            break
        else:
            prefix_nodes.append(node)

    assert len(prefix_nodes) > 0, "Prefix nodes must not be empty"

    for node in prefix_nodes:
        new_node = prefix_graph.node_copy(node, lambda n: value_remap[n])
        value_remap[node] = new_node

    last_node = prefix_nodes[-1]
    prefix_graph.output(value_remap[last_node])

    prefix_gm = fx.GraphModule(root=graph_module, graph=prefix_graph)
    return prefix_gm, value_remap


def get_fx_submodule(graph_module, value_remap, start_node, end_node):
    """
    Extracts a middle subgraph from an FX GraphModule between `start_node` and `end_node`.

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
    assert isinstance(graph_module, fx.GraphModule)
    assert isinstance(value_remap, dict)
    assert (
        len(value_remap) > 0
    ), "Remap dict cant be empty for slices in the middle of the model"
    graph = graph_module.graph
    new_nodes = []
    new_graph = fx.Graph()
    keep = False

    for node in graph.nodes:
        if node.name == start_node:
            keep = True
        if node.name == end_node:
            break
        if keep:
            new_nodes.append(node)

    assert len(new_nodes) > 0, "Node list must not be empty"

    # Adds placeholder to the beginning of subgraph so that its forward can take an input
    first_node = new_nodes[0]
    for arg in first_node.args:
        if isinstance(arg, fx.Node):
            ph = new_graph.placeholder(f"input_{arg.name}")
            value_remap[arg] = ph

    for node in new_nodes:
        new_node = new_graph.node_copy(node, lambda n: value_remap[n])
        value_remap[node] = new_node

    last_node = new_nodes[-1]
    new_graph.output(value_remap[last_node])

    new_gm = fx.GraphModule(root=graph_module, graph=new_graph)

    return new_gm, value_remap


def get_suffix_submodule(
    graph_module: fx.GraphModule, value_remap: dict, start_node: str
):
    """
    Extracts the subgraph from `start_node` (inclusive) to the final output of the model.

    Parameters
    ----------
    graph_module : fx.GraphModule
        The FX-traced full model.
    value_remap : dict
        Mapping from previous nodes to their placeholders/substitutes (for start_node input).
    start_node : str
        Name of the node where the suffix begins.

    Returns
    -------
    suffix_gm : fx.GraphModule
        FX GraphModule for the suffix.
    value_remap : dict
        Updated mapping including suffix nodes.
    """
    graph = graph_module.graph
    new_graph = fx.Graph()
    new_nodes = []
    keep = False

    for node in graph.nodes:
        if node.name == start_node:
            keep = True
        if keep:
            new_nodes.append(node)

    assert len(new_nodes) > 0, "Suffix nodes cannot be empty"

    # Adds placeholder to the beginning of subgraph so that its forward can take an input
    first_node = new_nodes[0]
    for arg in first_node.args:
        if isinstance(arg, fx.Node):
            ph = new_graph.placeholder(f"input_{arg.name}")
            value_remap[arg] = ph

    for node in new_nodes:
        new_node = new_graph.node_copy(node, lambda n: value_remap[n])
        value_remap[node] = new_node

    # Explicitly define output of the subgraph
    last_node = new_nodes[-1]
    new_graph.output(value_remap[last_node])

    suffix_gm = fx.GraphModule(graph_module, new_graph)
    return suffix_gm, value_remap


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
    ), "Graph module must be an instance of fx.GraphModule"
    assert len(prune_modules_name) > 0, "Prune list must not be empty"
    prune_subnets = []
    dense_subnets = []
    remap = {}

    gm = graph_module

    for idx, name in enumerate(prune_modules_name):
        fx_name = "_".join(name.split("."))

        if idx == 0:
            subnet, remap = get_initial_prefix_submodule(
                graph_module=gm, end_node=fx_name
            )
        elif idx == len(prune_modules_name) - 1:
            subnet, remap = get_suffix_submodule(
                graph_module=gm, value_remap=remap, start_node=fx_name
            )
        else:
            end_node = "_".join(prune_modules_name[idx + 1].split("."))
            subnet, remap = get_fx_submodule(
                graph_module=gm,
                value_remap=remap,
                start_node=fx_name,
                end_node=end_node,
            )

        prune_subnets.append(subnet)
        dense_subnets.append(subnet)

    return prune_subnets, dense_subnets


def perform_local_search(
    w_optimal,
    layer,
    p,
    prune_by_iter: Optional[list],
    sym_diff_per_iter: Optional[list],
    prune_per_iter=2,
):
    assert isinstance(layer, nn.Conv2d)
    assert isinstance(p, int) and p > 0
    if prune_by_iter is None:
        prune_list = []
        num_iterations = p // prune_per_iter
        rem = p % prune_per_iter
        zero_set = set()

        if p >= prune_per_iter:
            prune_list.extend([prune_per_iter for i in range(num_iterations)])
        if rem > 0:
            prune_list.extend([rem])
    else:

        assert np.sum(np.array(prune_by_iter)) == p
        prune_list = prune_by_iter

    if sym_diff_per_iter is not None:
        assert len(prune_list) == len(sym_diff_per_iter)
        assert all(p <= t for p, t in zip(prune_list, sym_diff_per_iter))

    else:
        sym_diff_per_iter = prune_list

    # for i in range(len(sym_diff_per_iter)):
    #     num_prune_iter = sym_diff_per_iter[]


def osscar_prune(
    dense_subnet,
    prune_subnet,
    dense_input,
    pruned_input,
    cached_dense_out,
    cached_prune_out,
    layer_name,
    prune_layer_counts,
):
    assert isinstance(dense_subnet, nn.Module), "Needs a valid subnet(nn.Module)"
    assert isinstance(prune_subnet, nn.Module), "Needs a valid subnet(nn.Module)"
    assert isinstance(dense_input, torch.Tensor)
    assert isinstance(pruned_input, torch.Tensor)
    assert pruned_input.ndim == 4, "Pruned input tensor must be of the shape B, C, H, W"
    assert dense_input.ndim == 4, "Dense input tensor must be of the shape B, C, H, W"

    num_batches = dense_input.shape[0]
    conv_module = None
    reshaped_conv_wt = None

    # The way subnets are designed nmakes the first module of a subnet the conv2d thats needed to be pruned
    for _, module in dense_subnet.named_modules():
        conv_module = module
        assert isinstance(conv_module, nn.Conv2d)
        reshaped_conv_wt = reshape_filter(conv_module.weight)
        break

    print(conv_module)

    assert (
        num_batches == pruned_input.shape[0]
    ), "Dense and prune inputs must have the same number of images"

    total_xtx = 0
    total_xty = 0

    for batch_idx in range(num_batches):
        dense_batch = dense_input[batch_idx, :, :, :]
        pruned_batch = pruned_input[batch_idx, :, :, :]
        dense_batch = reshape_conv_layer_input(input=dense_batch, layer=conv_module)
        pruned_batch = reshape_conv_layer_input(input=pruned_batch, layer=conv_module)
        total_xtx += get_coeff_h(pruned_batch)
        total_xty += get_XtY(pruned_batch, dense_batch)

    # Note to self: Remember to double check
    total_xtx /= num_batches
    total_xty /= num_batches

    w_optimal = get_optimal_W(
        gram_xx=total_xtx, gram_xy=total_xty, dense_weights=reshaped_conv_wt
    )

    # Note to self: prune_layer_counts is supposed to be a dicttionary now, beware, dont forget to change
    p = prune_layer_counts["layer_name"]
    # pruned_layer = perform_local_search(
    #     w_optimal, layer_name, prune_layer_counts
    # )

    # Perform forward till the next layer to be pruned and cache results
    # dense_out = dense_subnet(dense_input[batch_idx, :, :, :])
    # prune_out = prune_subnet(pruned_input[batch_idx, :, :, :])

    # cached_out =


if __name__ == "__main__":
    model = resnet50(pretrained=True)
    input = torch.randn(1, 3, 224, 224)

    weights = model.conv1.weight
    # print(weights.shape)

    # model.conv1.weight = weights[:, :2, :, :]

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

    print(prune_subnets[0])
    print(dense_subnets[1])
    # prefix_gm, remap = get_initial_prefix_submodule(graph_module=gm, end_node=end)

    # start_node = reformatted[0]
    # end_node = reformatted[1]

    # subnet, remap_dict = get_fx_submodule(
    #     graph_module=gm, value_remap=remap, start_node=start_node, end_node=end_node
    # )

    # out = prefix_gm(input)
    # # print(out.shape)

    # # out = out[:, :2, :, :]
    # # model.conv1.weight = torch.nn.Parameter(weights[:, :2, :, :], requires_grad=True)

    # out = subnet(out)

    # print(subnet.conv1)

    # print(out.shape)

    # outputs = {}

    # def hook_fn(model, input, output):
    #     outputs["maxpool"] = output

    # model.maxpool.register_forward_hook(hook_fn)
    # model(input)
    # print(outputs["maxpool"].shape)

    # suffix_module, remap_final = get_suffix_submodule(
    #     graph_module=gm, value_remap=remap, start_node=end_node
    # )
    # out = suffix_module(out)
    # # print(suffix_module)

    # direct_out = model(input)

    # assert torch.allclose(out, direct_out)
    # print(out.shape)
    # print(direct_out.shape)

    # perform_local_search()
    p = 1
    prune_per_iter = 3
    num_iterations = p // prune_per_iter
    rem = p % prune_per_iter
    zero_set = set()
    sym_diff_per_iter = []
    # while p > 1:
    #     p = p - prune_per_iter
    #     sym_diff_per_iter.extend([prune_per_iter])
    if p >= prune_per_iter:
        sym_diff_per_iter.extend([prune_per_iter for i in range(num_iterations)])
    if rem > 0:
        sym_diff_per_iter.extend([rem])
    print(sym_diff_per_iter)
    # for
