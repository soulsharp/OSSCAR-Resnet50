from typing import Optional
from itertools import chain

import numpy as np
import torch
import torch.fx as fx
from torch import nn

from compress.heuristics import collect_convolution_layers_to_prune
from compress.osscar_utils import get_external_nodes

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


def get_initial_prefix_submodule(gm, end_node, external_nodes):
    """
    Extracts a prefix subgraph from the start of an FX `GraphModule` up to
    (but not including) the node named `end_node`.

    Parameters
    ----------
    gm : torch.fx.GraphModule
        The FX-traced module whose graph will be sliced.
    end_node : str
        The name of the node where prefix extraction should stop. All nodes
        before this in execution order are included; `end_node` itself is excluded.
    external_nodes : dict
        Dictionary whose values are iterables of node names that represent
        external dependencies for downstream subgraphs. Any prefix node whose
        name appears here is exposed as an additional output.

    Returns
    -------
    prefix_gm : torch.fx.GraphModule
        A new GraphModule containing the prefix portion of the original graph.
        If external dependencies are present, the module outputs a dictionary
        mapping dependency names (plus `"output"`) to their corresponding values;
        otherwise it outputs the last prefix node directly.

    Notes
    -----
    - External dependencies referenced inside the prefix become placeholders
      named ``external__<node_name>``.
    - The main output of the prefix is the last node before `end_node`.
    - The returned GraphModule behaves like a regular PyTorch module.
    """
    assert isinstance(external_nodes, dict), "External nodes must be a dictionary"
    assert isinstance(gm, fx.GraphModule)
    prefix_graph = fx.Graph()
    env = {}
    external_deps = set(external_nodes.values())
    external_deps_names = set(node.name for node in external_deps)
    out_dict = {}
    last_node = None

    def fetch_arg(n):
        # If literal (non-Node), return as-is
        if not isinstance(n, fx.Node):
            return n

        # already copied
        if n in env:
            return env[n]

        ph = prefix_graph.placeholder(f"external__{n.name}")
        env[n] = ph
        return ph

    for node in gm.graph.nodes:
        if node.name == end_node:
            break
        else:
            new_node = prefix_graph.node_copy(node, fetch_arg)
            env[node] = new_node
            last_node = new_node
            if node.name in external_deps_names:
                out_dict[node.name] = new_node

    # Explicitly adds output node
    if out_dict:
        out_dict["output"] = last_node
        prefix_graph.output({k: v for k, v in out_dict.items()})
    else:
        prefix_graph.output(last_node)

    prefix_gm = fx.GraphModule(root=gm, graph=prefix_graph)
    return prefix_gm


def get_fx_submodule(gm, start_node, end_node, external_nodes):
    """
    Extracts a contiguous subgraph from an FX `GraphModule`, starting at
    `start_node` and ending just before `end_node`.

    Parameters
    ----------
    gm : torch.fx.GraphModule
        The FX-traced module whose graph will be sliced.
    start_node : str
        Name of the node where extraction should begin (inclusive).
    end_node : str
        Name of the node where extraction should stop (exclusive).
    external_nodes : dict
        Dictionary whose values list node names that represent external
        dependencies required by downstream subgraphs. These nodes are
        exposed as additional outputs.

    Returns
    -------
    subnet_gm : torch.fx.GraphModule
        A new GraphModule containing the selected subgraph. If the slice
        has external dependencies, the output is a dictionary mapping
        dependency names plus `"output"` to their values; otherwise the
        output is the final node in the slice.

    Notes
    -----
    - Any argument node not already copied into the subgraph becomes a
      placeholder named ``external__<node_name>``.
    - The main output corresponds to the last node before `end_node`.
    """
    assert isinstance(external_nodes, dict), "External nodes must be a dictionary"
    graph = gm.graph
    new_graph = fx.Graph()
    env = {}
    last_new_node = None
    external_deps = set(external_nodes.values())
    external_deps_names = set(node.name for node in external_deps)
    out_dict = {}

    def fetch_arg(n):
        # If not fx Node, return
        if not isinstance(n, fx.Node):
            return n

        # already copied
        if n in env:
            return env[n]

        # External dependencies
        ph = new_graph.placeholder(f"external__{n.name}")
        env[n] = ph
        return ph

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
        last_new_node = new_node
        if node.name in external_deps_names:
            out_dict[f"external__{node.name}"] = new_node

    # Explicitly add output node(s)
    if last_new_node is None:
        raise RuntimeError("No nodes were copied: check start_node/end_node names")

    if out_dict:
        out_dict["output"] = last_new_node
        new_graph.output({k: v for k, v in out_dict.items()})
    else:
        new_graph.output(last_new_node)

    subnet_gm = fx.GraphModule(gm, new_graph)
    return subnet_gm


def get_suffix_submodule(gm, start_node, external_nodes):
    """
    Extracts the suffix subgraph of an FX `GraphModule` beginning at
    `start_node` and continuing through the model's final output.

    Parameters
    ----------
    gm : torch.fx.GraphModule
        The FX-traced module whose graph will be sliced.
    start_node : str
        Name of the node where the suffix begins (inclusive).
    external_nodes : dict
        Dictionary whose values list node names that represent external
        dependencies needed by this suffix. Such nodes become input
        placeholders in the constructed subgraph.

    Returns
    -------
    suffix_gm : torch.fx.GraphModule
        A new GraphModule containing the suffix portion of the graph.
        The output is the model's final node as replicated in the suffix.

    Notes
    -----
    - Any argument not already present in the slice is turned into a
      placeholder named ``external__<node_name>``.
    - The slice runs from `start_node` to the original graph's terminal
      output node.
    """
    assert isinstance(external_nodes, dict), "External nodes must be a dictionary"
    graph = gm.graph
    new_graph = fx.Graph()
    env = {}
    last_node = None
    external_deps = set(external_nodes.values())
    external_deps_names = set(node.name for node in external_deps)

    def fetch_arg(n):
        # For non-Node literals
        if not isinstance(n, fx.Node):
            return n

        # ALready exists in env
        if n in env:
            return env[n]

        # Create a placeholder for an external dependency
        ph = new_graph.placeholder(f"external__{n.name}")
        env[n] = ph
        return ph

    copying = False
    for node in graph.nodes:
        if node.name == start_node:
            copying = True

        if not copying:
            continue

        new_node = new_graph.node_copy(node, fetch_arg)
        env[node] = new_node
        last_node = new_node

    # Explicitly adds output node
    new_graph.output(last_node)

    suffix_gm = fx.GraphModule(gm, new_graph)
    return suffix_gm


def get_all_subnets(prune_modules_name, gm, external_nodes):
    """
    Constructs the prefix, middle, and suffix subgraphs around the modules
    specified for pruning, producing both pruned and dense (unmodified)
    versions of each slice.

    Parameters
    ----------
    prune_modules_name : list[str]
        List of module names in `graph_module` that will be pruned, in the
        order they appear in the model.
    gm : fx.GraphModule
        The full FX-traced model.
    external_nodes : dict
        Mapping of : node in present subraph -> upstream dependencies

    Returns
    -------
    prune_subnets : list[fx.GraphModule]
        Subgraphs corresponding to each slice that will undergo pruning.
    dense_subnets : list[fx.GraphModule]
        Matching subgraphs built from the original model, used as fixed
        references during pruning.

    Notes
    -----
    - The returned lists contain: (1) a shared prefix slice, (2) one slice
      per prune module, and (3) the final suffix slice.
    - Each slice is constructed twice to ensure pruned and dense versions
      are independent and do not share graph state.
    """
    assert isinstance(
        gm, fx.GraphModule
    ), "graph_module must be an instance of fx.GraphModule"
    assert len(prune_modules_name) > 0, "Prune list must not be empty"
    prune_subnets = []
    dense_subnets = []

    first_node_name = "_".join(prune_modules_name[0].split("."))
    prefix_subnet_dense = get_initial_prefix_submodule(
        gm=gm, end_node=first_node_name, external_nodes=external_nodes
    )
    prefix_subnet_pruned = get_initial_prefix_submodule(
        gm=gm, end_node=first_node_name, external_nodes=external_nodes
    )
    prune_subnets.append(prefix_subnet_pruned)
    dense_subnets.append(prefix_subnet_dense)

    for idx, name in enumerate(prune_modules_name):
        fx_name = "_".join(name.split("."))

        if idx == len(prune_modules_name) - 1:
            dense_subnet = get_suffix_submodule(
                gm=gm, start_node=fx_name, external_nodes=external_nodes
            )
            pruned_subnet = get_suffix_submodule(
                gm=gm, start_node=fx_name, external_nodes=external_nodes
            )
        else:
            end_node = "_".join(prune_modules_name[idx + 1].split("."))
            dense_subnet = get_fx_submodule(
                gm=gm,
                start_node=fx_name,
                end_node=end_node,
                external_nodes=external_nodes,
            )
            pruned_subnet = get_fx_submodule(
                gm=gm,
                start_node=fx_name,
                end_node=end_node,
                external_nodes=external_nodes,
            )

        prune_subnets.append(pruned_subnet)
        dense_subnets.append(dense_subnet)

    return prune_subnets, dense_subnets


def display_subnet_info(subnet: fx.GraphModule):
    for node in subnet.graph.nodes:
        print("Node_name: ", node.name)
        print("Node_op:", node.op)
        print("Node_args: ", node.args)
        print("Node_kwargs: ", node.kwargs)

