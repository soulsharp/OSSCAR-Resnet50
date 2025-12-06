import random
from typing import cast, Optional
from itertools import chain

import numpy as np
import torch
from torch import nn

from compress.heuristics import *
from data.load_data import build_eval_dataloader, build_eval_dataset
from model.resnet import resnet50


def reshape_filter(filter):
    """
    Rearrange a Conv2d weight tensor according to OSSCAR paper.

    Parameters
    ----------
    filter : torch.Tensor
        Weight tensor of shape (C_out, C_in, K_h, K_w).

    Returns
    -------
    torch.Tensor
        Reshaped tensor of shape (C_in*K_h*K_w, C_out).
    """
    assert isinstance(filter, torch.Tensor)
    assert filter.ndim == 4, "Filter shape must be (Cout, Cin, Kh, Kw)"
    cout, _, _, _ = filter.size()
    reshaped_filter = filter.permute(1, 2, 3, 0)
    reshaped_filter = reshaped_filter.reshape(-1, cout)

    return reshaped_filter


def reshape_conv_layer_input(input, layer):
    """
    Unfold an input tensor using a Conv2d layer's settings.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape (C, H, W) or (B, C, H, W).
    layer : nn.Conv2d
        Conv2d layer whose kernel/stride/dilation/padding define the unfolding.

    Returns
    -------
    torch.Tensor
        Unfolded tensor of shape (B*L, C_in*K_h*K_w),
        where L is the number of sliding locations.
    """
    assert isinstance(input, torch.Tensor), "Input must be a tensor"
    assert isinstance(layer, nn.Conv2d), "Layer must be a nn.Conv2d layer"
    assert (
        input.ndim == 3 or input.ndim == 4
    ), "Input tensors must be either (C, H, W) or (B, C, H, W)"

    if input.ndim == 3:
        input = input.unsqueeze(dim=0)

    _, _, h, w = input.shape

    # Effective size of a kernel changes in a dilated conv op
    k_eff_y = (layer.kernel_size[0] - 1) * layer.dilation[0] + 1
    k_eff_x = (layer.kernel_size[1] - 1) * layer.dilation[1] + 1

    if isinstance(layer.padding, str) and layer.padding == "same":
        y_padding = ((layer.stride[0] * h - h) + k_eff_y - layer.stride[0]) // 2
        x_padding = ((layer.stride[1] * w - w) + k_eff_x - layer.stride[1]) // 2
    elif isinstance(layer.padding, tuple):
        y_padding, x_padding = layer.padding
    else:
        y_padding = x_padding = layer.padding

    # Silence pylance's static check
    y_padding = cast(int, y_padding)
    x_padding = cast(int, x_padding)

    unfold = nn.Unfold(
        kernel_size=layer.kernel_size,
        dilation=layer.dilation,
        padding=(y_padding, x_padding),
        stride=layer.stride,
    )

    input = unfold(input)
    input = input.permute(1, 0, 2)
    input = input.flatten(1)
    input = input.T

    return input


def get_coeff_h(design_matrix):
    """
    Compute the input autocorrelation (H) matrix from a 2D design matrix.

    Parameters
    ----------
    design_matrix : torch.Tensor
        2D tensor of shape (N, D), where N is the number of samples
        (e.g., unfolded spatial positions across the batch) and
        D is the feature dimension (e.g., C_in * K_h * K_w).

    Returns
    -------
    torch.Tensor
        Square tensor of shape (D, D) representing the autocorrelation
        matrix H = XᵀX of the input features.
    """
    assert isinstance(design_matrix, torch.Tensor)
    assert design_matrix.ndim == 2, "Requires the reshaped design matrix"

    return design_matrix.T @ design_matrix


def get_XtY(X, Y):
    """
    Compute the cross-correlation matrix XᵀY between two unfolded tensors.

    Parameters
    ----------
    X : torch.Tensor
        2D tensor of shape (N, D₁), an unfolded input/design matrix
        from `reshape_conv_layer_input`, where N is the number of samples
        (e.g., batch x sliding locations) and D₁ is the feature dimension.
    Y : torch.Tensor
        2D tensor of shape (N, D₂), another unfolded tensor of
        the same number of rows as `X`, but possibly with a different
        feature dimension D₂.

    Returns
    -------
    torch.Tensor
        Matrix of shape (D₁, D₂) representing the cross-correlation
        XᵀY between the two inputs.
    """
    assert isinstance(X, torch.Tensor)
    assert isinstance(Y, torch.Tensor)

    A = X.T

    assert A.shape[1] == Y.shape[0]
    return A @ Y


def num_params_in_prune_channels(layers):
    """
    Compute the total number of parameters across a list of convolutional layers.

    Parameters
    ----------
    layers : list of nn.Conv2d
        The convolutional layers whose parameters you want to count.

    Returns
    -------
    int
        Total number of parameters (weights + biases if present) in the given layers.
    """
    params = 0

    for layer in layers:
        assert isinstance(layer, nn.Conv2d)
        params += count_parameters(layer, in_millions=False)

    return params


def recalculate_importance(rem_channels_layer_wise):
    """
    Recalculate normalized importance weights for each layer
    based on the remaining number of channels per layer.

    Parameters
    ----------
    rem_channels_layer_wise : array-like of int
        Number of channels remaining in each layer.

    Returns
    -------
    numpy.ndarray
        Normalized importance values for each layer (sum to 1).
    """
    total = np.sum(rem_channels_layer_wise)
    rem_imp = np.divide(rem_channels_layer_wise, total)

    return rem_imp


def distribute_remaining_parameters(
    rem_params, rem_channels_per_layer, layers, num_iters=20, allowable_tol=250
):
    """
    Stochastically allocate leftover parameter removals across layers.

    At each iteration, a layer is sampled according to a probability
    distribution proportional to its remaining channels. One input channel
    is removed from the chosen layer if possible, and the remaining parameter
    budget is updated. The loop stops when the budget is within the allowable
    tolerance or after `num_iters` iterations.

    Parameters
    ----------
    rem_params : int
        Remaining number of parameters to remove.
    rem_channels_per_layer : list of int
        Remaining number of channels per layer (mutable, will be updated).
    layers : list of nn.Conv2d
        Convolutional layers eligible for pruning.
    num_iters : int, optional
        Maximum number of allocation iterations. Default is 20.
    allowable_tol : int, optional
        Stop when the remaining parameter budget is within this tolerance. Default is 250.

    Returns
    -------
    tuple
        p : numpy.ndarray
            Array of additional channels removed per layer.
        rem_params : int
            Remaining number of parameters still to remove after allocation.
    """
    num_layers = len(layers)
    layer_choices = np.arange(num_layers)
    p = np.zeros(num_layers, dtype=np.int32)
    rng = random.Random(3)

    rem_imp = recalculate_importance(rem_channels_layer_wise=rem_channels_per_layer)

    for i in range(num_iters):
        random_layer_idx = rng.choices(layer_choices, weights=rem_imp)[0]
        assert isinstance(random_layer_idx, (int, np.integer))

        layer = layers[random_layer_idx]

        assert isinstance(layer, nn.Conv2d)
        params_removed = (
            layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        )
        count_remove_params = rem_params - params_removed
        if rem_channels_per_layer[random_layer_idx] > 1 and count_remove_params >= 0:
            p[random_layer_idx] += 1
            rem_channels_per_layer[random_layer_idx] -= 1

            if rem_params - count_remove_params < 0:
                continue

            rem_params = count_remove_params
            rem_imp = recalculate_importance(rem_channels_per_layer)

        if rem_params <= allowable_tol:
            break

    return p, rem_params


def get_count_prune_channels(model, prune_percentage, allowable_tol=250):
    """
    Compute per-layer channel pruning counts to reach a target global prune percentage.

    This function:
      1. Collects the convolutional layers eligible for pruning.
      2. Computes how many parameters to remove from each layer based on an
         importance heuristic.
      3. Converts parameter removals into channel counts per layer.
      4. Distributes any remaining parameter removals stochastically to meet
         the target budget within the allowable tolerance.

    Parameters
    ----------
    model : nn.Module
        The model to prune.
    prune_percentage : float
        Target fraction of total model parameters to remove (0-1).
    allowable_tol : int, optional
        Tolerance for how far from the target parameter count to allow. Default is 250.

    Returns
    -------
    tuple
        num_channels_left_per_layer : list of int
            Number of input channels to keep per layer after pruning.
        p : list or numpy.ndarray
            Number of input channels to prune per layer (sum of deterministic and random).
        remaining_params_to_prune : int
            Number of parameters still to prune after allocation (ideally <= allowable_tol).
    """
    model_total_params = count_parameters(model, in_millions=False)

    # Collect layers eligible to be pruned
    layers, _ = collect_convolution_layers_to_prune(model)

    # Quick sanity check : ensures that total params to be removed doesnt exceed those that can be provided by eligible layers
    eligible_prune_params_count = num_params_in_prune_channels(layers=layers)
    total_params_to_prune = int(model_total_params * prune_percentage)

    assert eligible_prune_params_count > total_params_to_prune

    # Computes relative importance of each layer, higher importance => More channels pruned from this layer
    importance_list = compute_layer_importance_heuristic(layers)

    assert math.isclose(
        np.sum(importance_list), 1.0, abs_tol=0.00001
    ), "importance scores must sum to 1"

    # Number of params to remove from every eligible layer
    num_prune_params_by_layer = total_params_to_prune * importance_list
    num_prune_params_by_layer = np.floor(total_params_to_prune * importance_list)

    revised_prune_params_count = 0
    p = []
    num_channels_left_per_layer = []

    for idx, layer in enumerate(layers):
        assert isinstance(layer, nn.Conv2d)

        num_spatial_params = layer.kernel_size[0] * layer.kernel_size[1]
        num_channels_per_filter = layer.in_channels
        num_filters = layer.out_channels

        # Num channels to remove from every player(defined as per osscar)
        num_channels_to_remove = num_prune_params_by_layer[idx] // (
            num_spatial_params * num_filters
        )
        p.append(num_channels_to_remove)

        assert (
            num_channels_to_remove < num_channels_per_filter
        ), "Cant remove all channels in a filter"

        num_params_removed = num_spatial_params * num_channels_to_remove * num_filters
        revised_prune_params_count += num_params_removed
        num_channels_left = num_channels_per_filter - num_channels_to_remove
        num_channels_left_per_layer.append(num_channels_left)

    remaining_params_to_prune = total_params_to_prune - revised_prune_params_count

    if remaining_params_to_prune > allowable_tol:
        p_rem, remaining_params_to_prune = distribute_remaining_parameters(
            rem_params=remaining_params_to_prune,
            rem_channels_per_layer=num_channels_left_per_layer,
            layers=layers,
            allowable_tol=allowable_tol,
        )

        p = np.array(p) + np.array(p_rem)

    return num_channels_left_per_layer, p, remaining_params_to_prune


def save_and_accumulate_layer_input(activations, layer_name):
    """
    Create a forward hook that computes and accumulates a per-layer statistic
    (e.g. Gram matrix) from the layer's input.

    Parameters
    ----------
    activations : dict
        A mutable dictionary where the accumulated statistic will be stored.
        Keys are layer names; values are initialized to 0 or an existing tensor.
    layer_name : str
        Key under which to accumulate this layer's statistic in `activations`.

    Returns
    -------
    hook : callable
        A forward hook with signature (module, input, output) suitable for
        passing to `module.register_forward_hook`. On each forward pass,
        it calls `get_coeff_h(design_matrix=input)` and adds the result to
        `activations[layer_name]`.

    Notes
    -----
    The hook expects the incoming `input` to be a 2-D tensor.
    """

    def hook(model, input, output):
        assert input.ndim == 2
        coeff_h = get_coeff_h(design_matrix=input)
        activations[layer_name] += coeff_h

    return hook


def register_hooks_to_collect_outs(prune_modules, prune_module_names, hook_fn):
    """
    Attach forward hooks to a list of Conv2d modules to accumulate per-layer statistics.

    Parameters
    ----------
    prune_modules : list[nn.Module]
        The modules (layers) to attach hooks to (e.g. Conv2d layers to be pruned).
    prune_module_names : list[str]
        Names corresponding to each module in `prune_modules`. Must match length.
    hook_fn : callable
        A factory that accepts (activations: dict, layer_name: str) and returns
        a forward hook function. For example, `save_and_accumulate_layer_input`.

    Returns
    -------
    activations : dict
        A dictionary keyed by `prune_module_names` whose values are initialized
        to 0. On each forward pass, the registered hooks will add to these values.

    Notes
    -----
    This utility is useful for computing and caching Gram matrices or other
    layer input statistics over a calibration dataset before pruning.
    """
    gram_activations = {name: 0.0 for name in prune_module_names}
    for idx, module in enumerate(prune_modules):
        assert isinstance(module, nn.Conv2d)
        module_name = prune_module_names[idx]
        module.register_forward_hook(
            hook=hook_fn(activations=gram_activations, layer_name=module_name)
        )

    return gram_activations


def recompute_X(prune_mask, X, activation_in_channels, kernel_height, kernel_width):
    """
    Recompute the unfolded input matrix X after pruning input channels.

    Parameters
    ----------
    prune_mask : array-like of bool
        Boolean mask of length `activation_in_channels` indicating which input channels to keep (`True`) or drop (`False`).
    X : torch.Tensor or np.ndarray
        2D unfolded input matrix of shape (N, M) where N corresponds to flattened channel-kernel elements and
        M to the number of sliding positions or samples.
    activation_in_channels : int
        Number of input channels in the convolution layer prior to pruning.
    kernel_height : int
        Height of the convolution kernel.
    kernel_width : int
        Width of the convolution kernel.

    Returns
    -------
    torch.Tensor or np.ndarray
        Pruned unfolded input matrix with columns corresponding only to kept input channels.
    """
    assert X.ndim == 2, "Input matrix must have already been reshaped"
    assert (
        len(prune_mask) == activation_in_channels
    ), "The length of the indicator vector and the number of in_channels must be the same"

    # Number of elements needed to represent one filter in_channel in the reshaped 2d weight matrix
    numel_one_channel = kernel_height * kernel_width
    _, X_width = X.shape

    slice_indices = np.arange(activation_in_channels)
    mask = np.ones(X_width, dtype=bool)
    slice_indices = [
        (start * numel_one_channel, start * numel_one_channel + numel_one_channel)
        for start in slice_indices
    ]

    for idx, indicator in enumerate(prune_mask):
        if not indicator:
            start, stop = slice_indices[idx]
            mask[start:stop] = False

    return X[:, mask]


def recompute_H(
    prune_mask,
    H,
    activation_in_channels,
    kernel_height,
    kernel_width,
    is_pure_gram=True,
):
    """
    Recompute the coefficient matrix H after pruning input channels.

    Parameters
    ----------
    prune_mask : array-like of bool
        Boolean mask of length `activation_in_channels` indicating which channels to keep (`True`) or drop (`False`).
    H : torch.Tensor or np.ndarray
        2D square matrix of shape (N, N) representing gram matrices over activations.
    kernel_height : int
        Height of the convolution kernel.
    kernel_width : int
        Width of the convolution kernel.
    activation_in_channels : int
        Number of input channels (activations) prior to pruning.

    Returns
    -------
    torch.Tensor or np.ndarray
        Pruned square matrix containing only rows and columns corresponding to kept input channels.
    """
    assert len(prune_mask) == activation_in_channels
    assert H.ndim == 2

    kept_indices = []
    numel_one_channel = kernel_height * kernel_width
    indices = np.arange(activation_in_channels)
    slice_indices = [
        (start * numel_one_channel, start * numel_one_channel + numel_one_channel)
        for start in indices
    ]
    if is_pure_gram:
        for idx, indicator in enumerate(prune_mask):
            if indicator:
                start, stop = slice_indices[idx]
                kept_indices.extend(np.arange(start=start, stop=stop))

        H_updated = H[np.ix_(kept_indices, kept_indices)]

        return H_updated

    else:
        H_width = H.shape[1]
        mask = np.ones(H_width, dtype=bool)
        # slice_indices = [
        #     (start * numel_one_channel, start * numel_one_channel + numel_one_channel)
        #     for start in slice_indices
        # ]

        for idx, indicator in enumerate(prune_mask):
            if not indicator:
                start, stop = slice_indices[idx]
                mask[start:stop] = False

        return H[:, mask]


def recompute_W(prune_mask, W, activation_in_channels, kernel_height, kernel_width):
    """
    Recompute the weight matrix W after pruning input channels.

    Parameters
    ----------
    prune_mask : array-like of bool
        Boolean mask of length `activation_in_channels` indicating which input channels to keep (`True`) or drop (`False`).
    W : torch.Tensor or np.ndarray
        2D weight matrix of shape (N, M) where N corresponds to flattened channel-kernel elements.
    activation_in_channels : int
        Number of input channels in the convolution layer prior to pruning.
    kernel_height : int
        Height of the convolution kernel.
    kernel_width : int
        Width of the convolution kernel.

    Returns
    -------
    torch.Tensor or np.ndarray
        Pruned weight matrix with rows corresponding only to kept input channels.
    """
    assert W.ndim == 2, "Weight matrix must have already been reshaped"
    assert (
        len(prune_mask) == activation_in_channels
    ), "The length of the indicator vector and the number of in_channels must be the same"

    # Number of elements needed to represent one filter in_channel in the reshaped 2d weight matrix
    numel_one_channel = kernel_height * kernel_width
    W_height, _ = W.shape

    slice_indices = np.arange(activation_in_channels)
    mask = np.ones(W_height, dtype=bool)

    slice_indices = [
        (start * numel_one_channel, start * numel_one_channel + numel_one_channel)
        for start in slice_indices
    ]

    for idx, indicator in enumerate(prune_mask):
        if not indicator:
            start, stop = slice_indices[idx]
            mask[start:stop] = False

    return W[mask, :]


def compute_X_via_cholesky(A, B, C, eig_fallback_tol=1e-12):
    """
    Compute X = A^{-1} @ (B @ C) assuming A is PSD (possibly singular).
    - Adds jitter to A (A_reg = A + jitter*I).
    - Attempts Cholesky. If it fails, falls back to eigendecomposition-based solve.
    Returns X with same dtype/device as inputs.
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "A must be square"
    Y = B @ C
    n = A.shape[0]
    device = A.device
    dtype = A.dtype
    jitter = 1e-6

    # Adpative scaling for proper regularization to treat ill-conditioned matrices
    scale = max(
        torch.trace(A).abs().item() / n,
        torch.linalg.matrix_norm(A, ord="fro").item() / (n**0.5),
        1.0,
    )
    lambda_j = jitter * scale
    A_reg = A + lambda_j * torch.eye(n, device=device, dtype=dtype)

    # Cholesky decomposition only if A_ref is positive definite(SPD)
    L, info = torch.linalg.cholesky_ex(A_reg)
    if info == 0:
        # LL^T X = Y  =>  L Z = Y, then L^T X = Z
        Z = torch.linalg.solve_triangular(L, Y, upper=False, left=True)
        X = torch.linalg.solve_triangular(L.transpose(-2, -1), Z, upper=True, left=True)
        return X

    # Fallback: eigen-decomposition for PSD
    eigvals, eigvecs = torch.linalg.eigh(A)
    reg_eig = eigvals + lambda_j
    # If reg_eig very small, clamp with tol
    reg_eig_clamped = torch.where(
        reg_eig.abs() < eig_fallback_tol,
        torch.full_like(reg_eig, eig_fallback_tol),
        reg_eig,
    )

    # X = V diag(1/reg_eig_clamped) V^T Y
    VtY = eigvecs.transpose(-2, -1) @ Y
    scaled = VtY / reg_eig_clamped.unsqueeze(-1)
    X = eigvecs @ scaled
    return X


def get_optimal_W(gram_xx, gram_xy, dense_weights):
    """
    Compute the optimal weight matrix for a layer in OSSCAR using precomputed Gram matrices.

    This function solves for W* in the least-squares sense:
        min_W ||X W - Y||_F^2
    where:
        - gram_xx = X^T X
        - gram_xy = X^T Y_dense
        - dense_weights = Y_dense reshaped as 2D matrix

    It handles potentially ill-conditioned or nearly singular Gram matrices using
    adaptive Tikhonov regularization. Depending on the conditioning, it either:
        1. Uses Cholesky decomposition if the matrix is SPD.
        2. Falls back to eigen-decomposition with clamping for positive semi-definite cases.

    Parameters
    ----------
    gram_xx : torch.Tensor, shape (N, N)
        The input Gram matrix X^T X. Must be square.
    gram_xy : torch.Tensor, shape (N, N)
        The cross Gram matrix X^T Y_dense. Must be square.
    dense_weights : torch.Tensor, shape (N, N)
        The target dense weight matrix reshaped to 2D.

    Returns
    -------
    torch.Tensor, shape (N, N)
        The optimal weight matrix W* for the current layer.

    Notes
    -----
    - Adds adaptive regularization proportional to the trace and Frobenius norm of `gram_xx`.
    - Uses `torch.linalg.cholesky_ex` to safely attempt Cholesky decomposition.
    - If Cholesky fails (matrix not SPD), falls back to eigen-decomposition and clamps small eigenvalues.
    - Intended to be called once per layer during the first OSSCAR iteration.

    Raises
    ------
    AssertionError
        If input tensors are not 2D, or if shapes do not match as expected.
    """

    assert isinstance(gram_xx, torch.Tensor)
    assert isinstance(gram_xy, torch.Tensor)
    assert isinstance(dense_weights, torch.Tensor)

    assert gram_xx.ndim == 2, "Gram matrix XtX should be 2D"
    assert gram_xy.ndim == 2, "Gram matrix XtY should be 2D"
    assert dense_weights.ndim == 2, "Weights should have been reshaped to 2D"
    assert gram_xx.shape[0] == gram_xx.shape[1], "Gram_xx must be square"
    assert gram_xy.shape[0] == gram_xy.shape[1], "Gram_xy must be square"

    Y = gram_xy @ dense_weights
    n = gram_xx.shape[0]
    device = gram_xx.device
    dtype = gram_xx.dtype
    jitter = 1e-6
    eig_fallback_tol = 1e-12

    # Adpative scaling for proper regularization to treat ill-conditioned matrices
    scale = max(
        torch.trace(gram_xx).abs().item() / n,
        torch.linalg.matrix_norm(gram_xx, ord="fro").item() / (n**0.5),
        1.0,
    )
    lambda_j = jitter * scale
    gram_xx_reg = gram_xx + lambda_j * torch.eye(n, device=device, dtype=dtype)

    # Cholesky decomposition only if gram_xx_reg is positive definite(SPD)
    L, info = torch.linalg.cholesky_ex(gram_xx_reg)
    if info == 0:
        # LL^T X = Y  =>  L Z = Y, then L^T X = Z
        Z = torch.linalg.solve_triangular(L, Y, upper=False, left=True)
        X = torch.linalg.solve_triangular(L.transpose(-2, -1), Z, upper=True, left=True)
        return X

    # Fallback: eigen-decomposition for PSD
    eigvals, eigvecs = torch.linalg.eigh(gram_xx)
    reg_eig = eigvals + lambda_j
    # If reg_eig very small, clamp with tol
    reg_eig_clamped = torch.where(
        reg_eig.abs() < eig_fallback_tol,
        torch.full_like(reg_eig, eig_fallback_tol),
        reg_eig,
    )

    # X = V diag(1/reg_eig_clamped) V^T Y
    VtY = eigvecs.transpose(-2, -1) @ Y
    scaled = VtY / reg_eig_clamped.unsqueeze(-1)
    X = eigvecs @ scaled
    return X


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
        ValueError: If the target module isn't found.
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

    print(f"Pruning layer : {layer._get_name}...")

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
    # Debugging note : Is the first named_module guaranteed to be a nn.Conv2d
    # even after the changes involving the placeholders?
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

    keep_mask, _, _ = perform_local_search(
        dense_weights=reshaped_conv_wt,
        layer=conv_module,
        p=layer_prune_channels,
        gram_xx=total_xtx,
        gram_xy=total_xty,
    )

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

    # Debugging note : Does replace_module still work aftet the inclusion of the placeholder
    # + residual connection logic?
    replace_module(
        model=pruned_subnet, target_module=conv_module, new_module=new_conv_module
    )

    return pruned_subnet, keep_mask


class DenseSubnetRunner(nn.Module):
    def __init__(self, subnet_gm: torch.fx.GraphModule, external_nodes: dict):
        super().__init__()
        self.gm = subnet_gm
        self.external_nodes = external_nodes
        self.upstream_deps = external_nodes.values()
        self.upstream_node_names = [node.name for node in self.upstream_deps]

    def get_prefix_removed_name(self, prefix_name, prefix="external__"):
        return prefix_name[len(prefix) :]

    def forward(self, x, ctx: dict, batch_idx: int = 0):
        """
        Execute subnet with `x` and `ctx`, returning output + updated `ctx`.
        """
        inputs = {}
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                node_name = node.name
                if self.get_prefix_removed_name(node_name) in self.upstream_node_names:
                    ctx_tensor = ctx[node_name]
                    inputs[node_name] = ctx_tensor[batch_idx]
                else:
                    inputs[node_name] = x

        # for k, t in inputs.items():
        #     if isinstance(t, torch.Tensor):
        #         print(f"Debse Runner input {k}: ndim={t.ndim}, shape={t.shape}")
        #     else:
        #         print(f"Dense Runner input {k}: type={type(t)} (BAD)")

        # print("Input keys : ", inputs.keys())
        out = self.gm(**inputs)

        if isinstance(out, dict):
            for k, v in out.items():
                if k != "output":
                    if k not in ctx:
                        ctx[k] = []
                    ctx[k].append(v)
            return out["output"], ctx

        return out, ctx


class SubnetRunner(nn.Module):
    def __init__(
        self,
        subnet_gm: torch.fx.GraphModule,
        external_nodes: dict,
        pruned_conv_target,
    ):
        """
        Runs a sliced FX subgraph and automatically supplies required inputs.

        Parameters
        ----------
        subnet_gm : torch.fx.GraphModule
            The sliced subgraph whose placeholders include:
            - One real input placeholder (receives `x`)
            - External placeholders (filled from `ctx`)
        external_nodes : dict
            Mapping: sliced_placeholder_name → original_node_name.
            Used to fetch upstream activations from `ctx` during execution.

        Notes
        -----
        - Placeholders not in `external_nodes.values()` get `x`.
        - External placeholders get `ctx[original_name]`.
        - Dict outputs store all keys except `"output"` into `ctx`.
        """
        super().__init__()
        self.gm = subnet_gm
        self.external_nodes = external_nodes
        self.upstream_deps = external_nodes.values()
        self.downstream_consumers = external_nodes.keys()
        print("External nodes: ", self.external_nodes)
        self.upstream_node_names = [node.name for node in self.upstream_deps]
        self.pruned_conv_target = pruned_conv_target
        print(f"Prune_target is {self.pruned_conv_target}")
        self.placeholder_mask_map = self.build_placeholder_mask_map()

    def build_placeholder_mask_map(self):
        """
        Returns:
            dict[str, bool] : placeholder_name -> should_mask
        """

        graph = self.gm.graph

        # Locate which node(conv2d layer) was pruned in this subnet
        pruned_node = None
        for node in graph.nodes:
            if node.op == "call_module" and node.name == self.pruned_conv_target:
                pruned_node = node
                break

        if pruned_node is None:
            raise RuntimeError(
                f"Pruned conv {self.pruned_conv_target} not found in subnet graph"
            )

        # Backward BFS - to locate all ancestors of the pruned_conv_target node
        visited = set()
        stack = [pruned_node]

        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)

            for arg in n.all_input_nodes:
                stack.append(arg)

        placeholder_mask_map = {}

        # Only masks
        for node in graph.nodes:
            if node.op == "placeholder":
                placeholder_mask_map[node.name] = node in visited

        return placeholder_mask_map

    def get_prefix_removed_name(self, prefix_name, prefix="external__"):
        return prefix_name[len(prefix) :]

    def forward(self, x, ctx: dict, batch_idx: int = 0, keep_mask: list = []):
        inputs = {}

        for node in self.gm.graph.nodes:
            if node.op != "placeholder":
                continue

            name = node.name

            # Real inputs
            if self.get_prefix_removed_name(name) not in self.upstream_node_names:
                inputs[name] = x
                continue

            # This placeholder comes from ctx
            tensor = ctx[name][batch_idx]

            # Mask if BFS says so
            if self.placeholder_mask_map.get(name, False):
                tensor = tensor[:, keep_mask, :, :]

            inputs[name] = tensor

        out = self.gm(**inputs)

        if isinstance(out, dict):
            for k, v in out.items():
                if k != "output":
                    ctx.setdefault(k, []).append(v)
            return out["output"], ctx

        return out, ctx


def is_real_consumer(node):
    # Ignores shape-only ops
    if node.op == "call_method" and node.target in [
        "size",
        "__getitem__",
        "reshape",
        "view",
    ]:
        return False
    return True


# External dependencies needed across subnet boundaries
def get_external_nodes(gm: torch.fx.GraphModule):
    external_nodes = {}
    for node in gm.graph.nodes:
        real_users = [u for u in node.users if is_real_consumer(u)]
        if len(real_users) > 1:
            for user in real_users:
                if "downsample" in str(user.target) or "add" in str(user.target):
                    external_nodes[user.name] = node

    return external_nodes
