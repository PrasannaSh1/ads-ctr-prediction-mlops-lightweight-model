"""Creates the optimizer named in hp."""
import torch


def make_optimizer(model, hp):
    if hp["optimizer"] == "adam":
        optimizer_class = torch.optim.Adam
    elif hp["optimizer"] == "adamw":
        optimizer_class = torch.optim.AdamW
    elif hp["optimizer"] == "adagrad":
        optimizer_class = torch.optim.Adagrad
    else:
        raise ValueError("unknown optimizer: " + str(hp["optimizer"]))
    return optimizer_class(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
