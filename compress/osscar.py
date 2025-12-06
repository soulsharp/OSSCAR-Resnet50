import argparse

import torch
import torch.nn as nn
import torch.fx as fx

from compress.graph_extractor import get_all_subnets
from compress.heuristics import collect_convolution_layers_to_prune
from compress.osscar_utils import (
    get_count_prune_channels,
    SubnetRunner,
    DenseSubnetRunner,
    is_real_consumer,
    get_external_nodes,
    prune_one_layer,
)
from data.load_data import build_calibration_dataloader, build_eval_dataset
from model.resnet import resnet50
from utils.utils import set_global_seed, safe_free, load_yaml


def run_forward_with_mask(
    subnet,
    input,
    input_mask=None,
    is_input_loader=False,
    external_nodes=None,
    ctx=None,
    prune_channel_name=None,
):
    """
    Run a subnet forward pass over batched inputs (tensor or DataLoader), optionally
    masking input channels. Uses SubnetRunner so external dependencies are pulled
    from `ctx`, which is updated with new intermediate outputs.

    Args:
        subnet (fx.GraphModule): Subnet to execute.
        input (Tensor | DataLoader): 5D tensor (num_batches, B, C, H, W) or loader.
        input_mask (BoolTensor, optional): Channel mask.
        is_input_loader (bool): Whether `input` is a DataLoader.
        external_nodes (dict): External dependency mapping.
        ctx (dict): Context shared across subnets.

    Returns:
        cached_input (Tensor): Stacked per-batch outputs.
        ctx (dict): Updated context.
    """
    assert external_nodes is not None, "external_nodes must exist even as an empty dict"
    assert ctx is not None, "Context dictionary cannot be None"
    calibration_batches = []
    if input_mask is None:
        subnet_runner = DenseSubnetRunner(
            subnet_gm=subnet, external_nodes=external_nodes
        )
    else:
        subnet_runner = SubnetRunner(
            subnet_gm=subnet,
            external_nodes=external_nodes,
            pruned_conv_target=prune_channel_name,
        )
    if is_input_loader:
        assert isinstance(input, torch.utils.data.DataLoader)
        for images, _ in input:
            outs, ctx = subnet_runner(images, ctx)
            calibration_batches.append(outs)
    else:
        assert isinstance(input, torch.Tensor)
        assert input.ndim == 5, "Input shape must be (Num_batches, Batch_size, C, H, W)"
        num_batches = input.shape[0]

        if input_mask is not None:
            masked_input = input[:, :, input_mask, :, :]
            keep_mask = input_mask
            for i in range(num_batches):
                current_batch = masked_input[i, :, :, :, :]
                output_tensor, ctx = subnet_runner(
                    x=current_batch, ctx=ctx, keep_mask=keep_mask, batch_idx=i
                )
                calibration_batches.append(output_tensor)
        else:
            masked_input = input
            for i in range(num_batches):
                current_batch = masked_input[i, :, :, :, :]
                output_tensor, ctx = subnet_runner(
                    x=current_batch, ctx=ctx, batch_idx=i
                )
                calibration_batches.append(output_tensor)

    cached_input = torch.stack(calibration_batches, dim=0)
    for k, v in ctx.items():
        if isinstance(v, list):
            ctx[k] = torch.stack(v, dim=0)

    return cached_input, ctx


def transform_prune_name(name):
    name = name.split(".")
    return "_".join(name)


def run_osscar(model, calibration_loader, args):
    """
    Run OSSCAR-style structured channel pruning on a model using a calibration set.

    The procedure:
      1. Compute how many channels to prune per Conv layer.
      2. Symbolically trace the model and split it into prefix/middle/suffix subnets.
      3. Run each subnet on calibration data to cache activations.
      4. Iteratively prune layers using a local greedy search.
      5. Recompute cached inputs after each pruning step.

    Args:
        model (nn.Module): Dense model to prune.
        calibration_loader (DataLoader): Data for activation calibration.
        args (Namespace): Must contain `prune_percentage`.

    Returns:
        pruned_model (list[nn.Module]): Ordered list of pruned subnets.
        keep_masks (list[BoolTensor]): Per-layer channel masks indicating which
            input channels were kept after pruning.
    """
    prune_percentage = args.prune_percentage
    _, prune_channels_by_layer, _ = get_count_prune_channels(
        model=model, prune_percentage=prune_percentage
    )
    # if "conv1" in prune_channels_by_layer:
    #     prune_channels_by_layer.remove("conv1")
    # print(prune_channels_by_layer)
    _, prune_modules_name = collect_convolution_layers_to_prune(model=model)
    # if "conv1" in prune_modules_name:
    #     prune_modules_name.remove("conv1")
    # print("Channels to  prune : ", prune_modules_name)

    gm = fx.symbolic_trace(model)
    external_nodes = get_external_nodes(gm)
    prune_subnets, dense_subnets = get_all_subnets(
        gm=gm, prune_modules_name=prune_modules_name, external_nodes=external_nodes
    )
    assert len(prune_subnets) == len(dense_subnets)

    keep_masks = []
    pruned_model = []
    prefix_subnet = prune_subnets[0]
    pruned_model.append(prefix_subnet)
    ctx = {}
    dense_ctx = {}

    dense_cached_input, dense_ctx = run_forward_with_mask(
        subnet=prefix_subnet,
        input=calibration_loader,
        input_mask=None,
        is_input_loader=True,
        external_nodes=external_nodes,
        ctx=dense_ctx,
    )
    cached_input = dense_cached_input.detach()

    for i in range(1, len(dense_subnets)):
        subnet_post_pruning, keep_mask = prune_one_layer(
            dense_subnet=dense_subnets[i],
            pruned_subnet=prune_subnets[i],
            dense_input=dense_cached_input,
            pruned_input=cached_input,
            layer_prune_channels=int(prune_channels_by_layer[i - 1]),
        )
        prune_channel_name = prune_modules_name[i - 1]
        prune_channel_name = transform_prune_name(prune_channel_name)
        print(f"Prune_channel_name = {prune_channel_name}")
        from compress.graph_extractor import display_subnet_info

        display_subnet_info(subnet_post_pruning)
        print(f"Keep mask after iter {i} : {keep_mask.sum()}")
        print(f"Shape of input to the dense forward : {dense_cached_input.shape}")
        pruned_model.append(subnet_post_pruning)
        keep_masks.append(keep_mask)
        new_dense, dense_ctx = run_forward_with_mask(
            subnet=dense_subnets[i],
            input=dense_cached_input,
            input_mask=None,
            external_nodes=external_nodes,
            ctx=dense_ctx,
        )
        print(f"Shape of dense_cached_input after iter {i} : {new_dense.shape}")
        ctx_vals = [(k, v.shape) for k, v in ctx.items()]
        print(f"Ctx vals shape after iter {i}: {ctx_vals}")
        print(f"Shape of input to prune forward : {cached_input.shape}")

        new_pruned, ctx = run_forward_with_mask(
            subnet=subnet_post_pruning,
            input=cached_input,
            input_mask=keep_mask,
            external_nodes=external_nodes,
            ctx=ctx,
            prune_channel_name=prune_channel_name,
        )
        print(
            f"Shape of cached_input after prune forward {i} times : {new_pruned.shape}"
        )
        print(f"Ctx after running prune forward {i} times : {ctx.keys()}")

        safe_free(dense_cached_input, cached_input)
        dense_cached_input, cached_input = new_dense.detach(), new_pruned.detach()

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
    parser = argparse.ArgumentParser(description="Arguments for OSSCAR")
    parser.add_argument("--prune_percentage", default=0.25, type=float)
    args = parser.parse_args()
    pruned_model_list, keep_mask_list = run_osscar(
        model=model, calibration_loader=calibration_dataloader, args=args
    )
    pruned_model = PrunedResnet50(pruned_model_list, keep_mask_list)
