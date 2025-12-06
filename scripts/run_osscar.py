import argparse

import torch

from data.load_data import build_calibration_dataloader, build_eval_dataset
from model.resnet import resnet50
from utils.utils import set_global_seed, load_yaml
from compress.osscar import run_osscar


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
    # parser.add_argument("--prune_layers", default=None, type=list, help="Provide names of layers to be pruned using OSSCAR. If None, the algorithm makes this decision by itself")
    args = parser.parse_args()
    pruned_model_list, keep_mask_list = run_osscar(
        model=model, calibration_loader=calibration_dataloader, args=args
    )
