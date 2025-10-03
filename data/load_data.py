import torch
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

from utils.utils import load_yaml


def build_train_dataset(cfg):
    """
    Creates the CIFAR-10 training dataset with random crop, horizontal flip, and normalization.

    Args:
        cfg (dict): Configuration containing:
            - "train_data_path" (str): Path to training data.
            - "mean" (list of float): Per-channel mean for normalization.
            - "std" (list of float): Per-channel std for normalization.

    Returns:
        torchvision.datasets.CIFAR10: Training dataset.
    """
    assert cfg is not None, "Config cannot be None"

    transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(cfg["mean"], cfg["std"]),
        ]
    )
    dataset = CIFAR10(
        root=cfg["train_data_path"], train=True, transform=transform, download=True
    )

    return dataset


def build_eval_dataset(cfg):
    """
    Creates the CIFAR-10 evaluation dataset with normalization.

    Args:
        cfg (dict): Configuration containing:
            - "test_data_path" (str): Path to evaluation data.
            - "mean" (list of float): Per-channel mean for normalization.
            - "std" (list of float): Per-channel std for normalization.

    Returns:
        torchvision.datasets.CIFAR10: Evaluation dataset.
    """
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(cfg["mean"], cfg["std"]),
        ]
    )
    dataset = CIFAR10(
        root=cfg["test_data_path"], train=False, transform=transform, download=True
    )

    return dataset


def build_calibration_dataloader(dataset, num_samples, batch_size=32):
    """
    Create a PyTorch DataLoader for a subset of a dataset to be used for calibration.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The full dataset to sample from.
    num_samples : int
        Number of samples from the start of the dataset to use for calibration.
    batch_size : int, optional (default=32)
        Batch size for the DataLoader.

    Returns
    -------
    calibration_dataloader : torch.utils.data.DataLoader
    """
    calibration_dataloader = torch.utils.data.DataLoader(
        dataset[:num_samples],
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    return calibration_dataloader


def build_train_dataloader(dataset, config):
    """
    Wraps a dataset in a DataLoader for training.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to wrap.
        config (dict): Configuration containing:
            - "train_batch_size" (int)
            - "num_workers" (int)
            - "pin_memory" (bool)

    Returns:
        torch.utils.data.DataLoader: DataLoader for training.
    """

    required_keys = ["train_batch_size", "num_workers", "pin_memory"]
    for key in required_keys:
        assert key in config, f"Missing key in config: {key}"

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        drop_last=True,
    )

    return train_loader


def build_eval_dataloader(dataset, config):
    """
    Wraps a dataset in a DataLoader for evaluation.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to wrap.
        config (dict): Configuration containing:
            - "val_batch_size" (int)
            - "num_workers" (int)
            - "pin_memory" (bool)

    Returns:
        torch.utils.data.DataLoader: DataLoader for evaluation/validation.
    """

    required_keys = ["val_batch_size", "num_workers", "pin_memory"]
    for key in required_keys:
        assert key in config, f"Missing key in config: {key}"

    # This dataloader is specifically meant for evaluation
    eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        drop_last=False,
    )

    return eval_loader


def prepare_dataloader(is_train):
    """
    Builds the training or evaluation DataLoader for CIFAR-10 conditioned on is_train flag.

    Returns:
        torch.utils.data.DataLoader -> A single dataloader conditioned on the value of is_train.
    """
    cfg = load_yaml("config/config.yaml")
    assert cfg is not None, "Config can't be None"

    if is_train:
        train_cfg = cfg["train"]
        train_dataset = build_train_dataset(train_cfg)
        loader = build_train_dataloader(train_dataset, train_cfg)

    else:
        eval_cfg = cfg["eval"]
        eval_dataset = build_eval_dataset(eval_cfg)
        loader = build_eval_dataloader(eval_dataset, eval_cfg)

    return loader
