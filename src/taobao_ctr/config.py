"""Reads params.yaml. Every setting of the project lives in that one file."""
import os

import yaml


def load_config(path="params.yaml"):
    """Read params.yaml and return it as a normal Python dictionary.

    Example: config["train"]["lr"] is the learning rate.
    Run every command from the project folder, so the file is found.
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def make_folders(config):
    """Create the data folder and the artifact folder (with its sub folders) if missing."""
    data_dir = config["paths"]["data_dir"]
    artifact_dir = config["paths"]["artifact_dir"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(os.path.join(artifact_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.join(artifact_dir, "preds"), exist_ok=True)
    os.makedirs(os.path.join(artifact_dir, "models"), exist_ok=True)
