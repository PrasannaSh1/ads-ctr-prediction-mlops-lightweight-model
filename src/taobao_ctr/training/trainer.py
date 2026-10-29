"""train_model(): best-checkpoint early stopping, predict(), metrics() (notebook section 7).

Checks validation AUC every ~eval_every_rows training rows; keeps the best checkpoint; stops
after `patience` checks without improvement, up to `max_epochs`. `report(i, auc)`, if given,
lets Optuna prune a trial (see training/tune.py) -- trainer.py itself has no Optuna dependency.
"""
from __future__ import annotations

import ctypes
import gc
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from taobao_ctr.config import Params
from taobao_ctr.data.dataset import Dataset
from taobao_ctr.training.batches import batches
from taobao_ctr.training.losses import make_loss
from taobao_ctr.training.optim import make_optimizer

HP_KEYS = ["lr", "batch_size", "dropout", "optimizer", "weight_decay", "loss", "gamma", "pos_weight", "cotrain_weight"]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def free_memory(device: torch.device | None = None) -> None:
    """Give unused RAM and GPU memory back."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # only available on glibc Linux (Colab/Kaggle)
    except OSError:
        pass
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()


def default_hp(params: Params) -> dict:
    """The DEFAULT_HP dict (notebook: `{k: CFG[k] for k in [...]}`), from params.yaml's `model`
    and `train` groups."""
    m, t = params.model, params.train
    return {
        "emb_dim": m.emb_dim, "tower_dim": m.tower_dim, "n_heads": m.n_heads, "n_query_tokens": m.n_query_tokens,
        "max_epochs": t.max_epochs, "eval_every_rows": t.eval_every_rows, "patience": t.patience,
        "lr": t.lr, "batch_size": t.batch_size, "dropout": t.dropout, "optimizer": t.optimizer,
        "weight_decay": t.weight_decay, "loss": t.loss, "gamma": t.gamma, "pos_weight": t.pos_weight,
        "cotrain_weight": t.cotrain_weight,
    }


@torch.no_grad()
def predict(model: torch.nn.Module, data: Dataset, rows: np.ndarray, device: torch.device, batch_size: int = 16384) -> dict:
    """Click probabilities of every head for the given rows."""
    model.eval()
    out = {}
    for b in batches(data, rows, batch_size, device):
        for k, v in model(b).items():
            out.setdefault(k, []).append(torch.sigmoid(v).float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def metrics(labels: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {"AUC": roc_auc_score(labels, p), "LogLoss": log_loss(labels, p), "PR_AUC": average_precision_score(labels, p)}


def train_model(
    model_cls, hp: dict, data: Dataset, fit_rows: np.ndarray, val_rows: np.ndarray,
    max_epochs: int, eval_every_rows: int, patience: int, device: torch.device, seed: int,
    report=None, verbose: bool = True,
):
    """Train, check validation AUC regularly, keep the best checkpoint."""
    set_seed(seed)
    model = model_cls(hp, data.sizes, data.seq_len).to(device)
    opt, loss_fn = make_optimizer(model, hp), make_loss(hp, device)
    eval_every = max(1, eval_every_rows // hp["batch_size"])
    best = {"auc": -1.0, "step": 0, "state": None}
    curve, step, bad, t0 = [], 0, 0, time.time()

    def check():
        nonlocal bad
        auc = roc_auc_score(data.y[val_rows], predict(model, data, val_rows, device)["logit"])
        model.train()
        curve.append((step, auc))
        if auc > best["auc"] + 1e-4:
            best.update(auc=auc, step=step, state={k: v.detach().clone() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
        if verbose:
            print(f"    step {step:>6}: val AUC {auc:.4f} (best {best['auc']:.4f}) | {(time.time() - t0) / 60:.1f} min")
        if report:
            report(len(curve), auc)  # may raise optuna.TrialPruned
        return bad >= patience

    model.train()
    stop = False
    for _epoch in range(max_epochs):
        for b in batches(data, fit_rows, hp["batch_size"], device, shuffle=True):
            out = model(b)
            loss = loss_fn(out["logit"], b["label"])
            if "cotrain_logit" in out:  # 3-tower: also train the cheap head
                loss = loss + hp["cotrain_weight"] * loss_fn(out["cotrain_logit"], b["label"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0 and check():
                stop = True
                break
        if stop:
            break
        if step % eval_every != 0 and check():  # also check at the end of every epoch
            break
    model.load_state_dict(best["state"])
    return model, best, curve
