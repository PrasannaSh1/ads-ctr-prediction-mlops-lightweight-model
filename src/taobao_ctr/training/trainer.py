"""Training loop, prediction and metrics for the deep models.

train_model() checks the validation AUC every eval_every_rows training rows, remembers the best
weights, and stops after `patience` checks without improvement (or after max_epochs).
"""
import gc
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from taobao_ctr.training.batches import batches
from taobao_ctr.training.losses import compute_loss
from taobao_ctr.training.optim import make_optimizer

# The settings Optuna is allowed to change
HP_KEYS = ["lr", "batch_size", "dropout", "optimizer", "weight_decay", "loss", "gamma", "pos_weight", "cotrain_weight"]


def set_seed(seed):
    """Same seed -> same random numbers -> the same result every run."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def free_memory():
    """Give unused RAM and GPU memory back."""
    gc.collect()
    torch.cuda.empty_cache()


def default_hp(config):
    """The default model + training settings, read from params.yaml."""
    m = config["model"]
    t = config["train"]
    hp = {
        "emb_dim": m["emb_dim"],
        "tower_dim": m["tower_dim"],
        "n_heads": m["n_heads"],
        "n_query_tokens": m["n_query_tokens"],
        "max_epochs": t["max_epochs"],
        "eval_every_rows": t["eval_every_rows"],
        "patience": t["patience"],
        "lr": t["lr"],
        "batch_size": t["batch_size"],
        "dropout": t["dropout"],
        "optimizer": t["optimizer"],
        "weight_decay": t["weight_decay"],
        "loss": t["loss"],
        "gamma": t["gamma"],
        "pos_weight": t["pos_weight"],
        "cotrain_weight": t["cotrain_weight"],
    }
    return hp


def predict(model, data, rows, device, batch_size=16384):
    """Click probabilities for the given rows. Returns a dict: head name -> numpy array."""
    model.eval()
    outputs = {}  # head name -> list of arrays, one per batch
    with torch.no_grad():
        for batch in batches(data, rows, batch_size, device):
            result = model(batch)
            for head_name, logits in result.items():
                probabilities = torch.sigmoid(logits).float().cpu().numpy()
                if head_name not in outputs:
                    outputs[head_name] = []
                outputs[head_name].append(probabilities)
    final = {}
    for head_name in outputs:
        final[head_name] = np.concatenate(outputs[head_name])
    return final


def metrics(labels, probabilities):
    """AUC, log loss and PR-AUC."""
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {
        "AUC": roc_auc_score(labels, probabilities),
        "LogLoss": log_loss(labels, probabilities),
        "PR_AUC": average_precision_score(labels, probabilities),
    }


def copy_weights(model):
    """A copy of all the model's weights, so we can go back to them later."""
    state = {}
    for name, value in model.state_dict().items():
        state[name] = value.detach().clone()
    return state


def check_validation(model, data, val_rows, device, step, best, curve, bad_checks, start_time, verbose, report):
    """Score the model on the validation rows and remember it if it is the best so far.

    Returns the new number of checks in a row without improvement.
    """
    probabilities = predict(model, data, val_rows, device)["logit"]
    auc = roc_auc_score(data.y[val_rows], probabilities)
    model.train()  # predict() switched the model to eval mode; switch back
    curve.append((step, auc))

    if auc > best["auc"] + 1e-4:
        best["auc"] = auc
        best["step"] = step
        best["state"] = copy_weights(model)
        bad_checks = 0
    else:
        bad_checks += 1

    if verbose:
        minutes = (time.time() - start_time) / 60
        print(f"    step {step:>6}: val AUC {auc:.4f} (best {best['auc']:.4f}) | {minutes:.1f} min")
    if report is not None:
        report(len(curve), auc)  # lets Optuna stop a bad trial early
    return bad_checks


def train_model(model_class, hp, data, fit_rows, val_rows, max_epochs, eval_every_rows, patience,
                device, seed, report=None, verbose=True):
    """Train a model and return (model with its best weights, best, curve)."""
    set_seed(seed)
    model = model_class(hp, data.sizes, data.seq_len).to(device)
    optimizer = make_optimizer(model, hp)
    eval_every = max(1, eval_every_rows // hp["batch_size"])  # check validation every this many batches

    best = {"auc": -1.0, "step": 0, "state": None}
    curve = []  # list of (step, validation AUC)
    step = 0
    bad_checks = 0
    start_time = time.time()
    stop = False

    model.train()
    for _epoch in range(max_epochs):
        for batch in batches(data, fit_rows, hp["batch_size"], device, shuffle=True):
            output = model(batch)
            loss = compute_loss(output["logit"], batch["label"], hp, device)
            if "cotrain_logit" in output:
                # The 3-tower model: also train its cheap head
                loss = loss + hp["cotrain_weight"] * compute_loss(output["cotrain_logit"], batch["label"], hp, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % eval_every == 0:
                bad_checks = check_validation(model, data, val_rows, device, step, best, curve, bad_checks,
                                              start_time, verbose, report)
                if bad_checks >= patience:
                    stop = True
                    break
        if stop:
            break
        if step % eval_every != 0:
            # Also check at the end of every epoch
            bad_checks = check_validation(model, data, val_rows, device, step, best, curve, bad_checks,
                                          start_time, verbose, report)
            if bad_checks >= patience:
                break

    model.load_state_dict(best["state"])  # go back to the best weights seen
    return model, best, curve
