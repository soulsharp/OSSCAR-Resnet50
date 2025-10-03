import torch
import yaml


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_yaml(path):
    """
    Loads a YAML file from the given path.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict or None: Parsed YAML contents, or None if loading fails.
    """
    try:
        with open(path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"File not found: {path}")
    except yaml.YAMLError as exc:
        print(f"YAML error: {exc}")
    return None


def count_parameters(model, in_millions=True):
    """
    Counts the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        float: Number of parameters (in millions).
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if in_millions:
        return params / 1000000
    else:
        return params


def get_topk_accuracy(logits, labels, k):
    """
    Computes the Top-K classification accuracy.

    Args:
        logits (torch.Tensor): Model output logits of shape (batch_size, num_classes).
        labels (torch.Tensor): Ground truth class labels of shape (batch_size,).
        k (int): The number of top predictions to consider (Top-K).

    Returns:
        float: Top-K accuracy as a fraction between 0 and 1.
    """
    assert (
        isinstance(logits, torch.Tensor) and logits.ndim == 2
    ), "Logits must be 2-dimensional tensors"
    assert (
        isinstance(labels, torch.Tensor) and labels.ndim == 1
    ), "Labels must be 1-dimensional tensors"
    assert isinstance(k, int) and k > 0, "K must be an integer greater than 0"

    # Top-k indices along classes
    topk_preds = torch.topk(logits, k, dim=1).indices
    labels = labels.view(-1, 1).expand_as(topk_preds)

    correct = (topk_preds == labels).any(dim=1).sum().item()
    total = labels.size(0)

    return correct / total


def return_train_val_cfg(path):
    cfg = load_yaml("path")
    assert cfg is not None
    train_cfg = cfg["train"]
    val_cfg = cfg["eval"]

    return train_cfg, val_cfg
