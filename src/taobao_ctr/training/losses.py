"""The loss function: binary cross-entropy (BCE), with two optional variants."""
import torch
import torch.nn.functional as F


def compute_loss(logits, labels, hp, device):
    """How wrong the predictions are. hp["loss"] picks one of three versions."""
    if hp["loss"] == "weighted_bce":
        # Clicks count pos_weight times more than non-clicks
        weight = torch.tensor(hp["pos_weight"], device=device)
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)

    if hp["loss"] == "focal":
        # Focal loss: easy examples count less, hard ones count more
        ce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        p = torch.sigmoid(logits)
        p_correct = p * labels + (1 - p) * (1 - labels)  # probability given to the true answer
        return ((1 - p_correct) ** hp["gamma"] * ce).mean()

    # Plain BCE
    return F.binary_cross_entropy_with_logits(logits, labels)
