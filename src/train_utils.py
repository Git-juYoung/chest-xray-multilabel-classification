import torch
import torch.nn as nn
import torch.optim as optim


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_criterion(pos_weight=None, device=None):
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        return nn.BCEWithLogitsLoss()

def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def build_optimizer(model, lr: float, weight_decay: float, use_adamw: bool = False):
    params = get_trainable_params(model)
    if use_adamw:
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, mode: str, factor: float, patience: int, verbose: bool = True):
     return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        verbose=verbose,
    )

