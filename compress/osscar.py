import argparse

import torch
import torch.nn as nn
import numpy as np
import torch.fx as fx

from compress.graph_extractor import get_all_subnets
from compress.heuristics import collect_convolution_layers_to_prune
from compress.osscar_utils import (
    get_count_prune_channels,
    is_real_consumer,
    get_external_nodes,
    prune_one_layer
)
from data.load_data import build_calibration_dataloader, build_eval_dataset
from model.resnet import resnet50
from utils.utils import set_global_seed, safe_free, load_yaml


def run_forward_with_mask(subnet, input, input_mask=None, is_input_loader=False):
    """
    Runs forward passes through a subnet, optionally masking input channels.

    Supports either a preloaded tensor of shape (num_batches, batch_size, C, H, W)
    or a DataLoader. The outputs from all batches are stacked along a new leading
    dimension for forming gram matrices.

    Args:
        subnet (nn.Module): The model or subnet to evaluate.
        input (torch.Tensor | DataLoader): Batched tensor input or a DataLoader.
        input_mask (torch.BoolTensor, optional): Channel-wise boolean mask; only
            channels with `True` are retained.
        is_input_loader (bool, default=False): If True, treats `input` as a DataLoader.

    Returns:
        cached_input (torch.Tensor): Stacked outputs for all batches, of shape
            (num_batches, batch_size, C_out, H_out, W_out).
    """
    calibration_batches = []
    if is_input_loader:
        assert isinstance(input, torch.utils.data.DataLoader)
        for images, _ in input:
            outs = subnet(images)
            calibration_batches.append(outs)
    else:
        assert isinstance(input, torch.Tensor)
        assert input.ndim == 5, "Input shape must be (Num_batches, Batch_size, C, H, W)"
        num_batches = input.shape[0]

        if input_mask is not None:
            input_tensor = input[:, :, input_mask, :, :]
        else:
            input_tensor = input

        # Note to self: could include all elements in 1 batch effectively making the input_tensor 4D
        # The for loop can then be avoided but might make the solution too memory-intensive
        for i in range(num_batches):
            current_batch = input_tensor[i, :, :, :, :]
            output_tensor = subnet(current_batch)
            calibration_batches.append(output_tensor)
    cached_input = torch.stack(calibration_batches, dim=0)

    return cached_input


def run_osscar(model, calibration_loader, args):
    """
    Apply OSSCAR-style structured pruning to a model using a calibration dataset.

    The function:
      1. Determines how many channels to prune per layer based on `args.prune_percentage`.
      2. Identifies convolution layers eligible for pruning.
      3. Symbolically traces the model to extract subnets for pruning.
      4. Iteratively prunes layers using local greedy search while caching intermediate activations.
      5. Returns a list of pruned subnets forming the pruned model and the per-layer keep masks.

    Parameters
    ----------
    model : nn.Module
        The original dense PyTorch model to prune.
    calibration_loader : torch.utils.data.DataLoader
        Data loader providing batches for calibration / activation caching.
    args : Namespace
        Arguments containing `prune_percentage` among other potential config values.

    Returns
    -------
    pruned_model : list[nn.Module]
        List of subnets forming the pruned model.
    keep_masks : list[torch.BoolTensor]
        Per-layer boolean masks indicating which input channels were kept.
    """

    prune_percentage = args.prune_percentage
    channels_post_prune, prune_channels_by_layer, remaining_params = (
        get_count_prune_channels(model=model, prune_percentage=prune_percentage)
    )
    # print(prune_channels_by_layer)
    _, prune_modules_name = collect_convolution_layers_to_prune(model=model)
    # print(prune_modules_name)

    gm = fx.symbolic_trace(model)
    prune_subnets, dense_subnets = get_all_subnets(
        graph_module=gm, prune_modules_name=prune_modules_name
    )
    assert len(prune_subnets) == len(dense_subnets)

    keep_masks = []
    pruned_model = []
    prefix_subnet = prune_subnets[0]
    pruned_model.append(prefix_subnet)

    dense_cached_input = run_forward_with_mask(
        subnet=prefix_subnet,
        input=calibration_loader,
        input_mask=None,
        is_input_loader=True,
    ).detach()
    cached_input = dense_cached_input

    # Bug: This code shouldn't execute when layer_prune_channels = 0
    for i in range(1, len(dense_subnets)):
        subnet_post_pruning, keep_mask = prune_one_layer(
            dense_subnet=dense_subnets[i],
            pruned_subnet=prune_subnets[i],
            dense_input=dense_cached_input,
            pruned_input=cached_input,
            layer_prune_channels=int(prune_channels_by_layer[i - 1]),
        )

        pruned_model.append(subnet_post_pruning)
        keep_masks.append(keep_mask)
        new_dense = run_forward_with_mask(
            subnet=dense_subnets[i], input=dense_cached_input, input_mask=None
        ).detach()
        new_pruned = run_forward_with_mask(
            subnet=subnet_post_pruning, input=cached_input, input_mask=keep_mask
        ).detach()

        safe_free(dense_cached_input, cached_input)
        dense_cached_input, cached_input = new_dense, new_pruned

    return pruned_model, keep_masks


class PrunedResnet50(nn.Module):
    def __init__(self, pruned_modules_list, keep_mask_list):
        super().__init__()
        assert len(pruned_modules_list) - 1 == len(keep_mask_list)
        self.module_list = nn.ModuleList(pruned_modules_list)

        # Register masks as buffers so that they can be saved/loaded along with the model
        for i, mask in enumerate(keep_mask_list):
            self.register_buffer(f"input_mask_{i}", mask)

        self.keep_mask_list = keep_mask_list

    def forward(self, x):
        # The first module has no input mask
        x = self.module_list[0](x)

        # The mask of the ith module in module_list is the i-1th entry of keep_mask_list
        for i in range(1, len(self.module_list)):
            x = x[:, self.keep_mask_list[i - 1], :, :]
            x = self.module_list[i](x)

        return x


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_global_seed(seed=232)
    g = torch.Generator()
    g.manual_seed(11)

    cfg_path = "config/config.yaml"
    cfg = load_yaml(cfg_path)
    assert cfg is not None, "Config cannot be empty or None"
    calibration_dataset = build_eval_dataset(cfg=cfg["eval"])
    calibration_dataloader = build_calibration_dataloader(
        dataset=calibration_dataset, num_samples=500, g=g, batch_size=32
    )

    model = resnet50(pretrained=True)
    # activation_in_channels = 3
    # numel_one_channel = 9
    # slice_indices = np.arange(activation_in_channels)
    # slice_indices = [
    #     (start * numel_one_channel, start * numel_one_channel + numel_one_channel)
    #     for start in slice_indices
    # ]
    # print(slice_indices)
    parser = argparse.ArgumentParser(description="Arguments for OSSCAR")
    parser.add_argument("--prune_percentage", default=0.25, type=float)
    args = parser.parse_args()
    pruned_model_list, keep_mask_list = run_osscar(
        model=model, calibration_loader=calibration_dataloader, args=args
    )
    pruned_model = PrunedResnet50(pruned_model_list, keep_mask_list)
