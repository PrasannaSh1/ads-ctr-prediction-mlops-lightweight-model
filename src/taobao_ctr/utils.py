"""Small helper functions used in more than one place."""
import re


def slug(text):
    """Turn a name into a safe id with only lowercase letters, digits and underscores.

    Example: "3-tower co-train" becomes "3_tower_co_train".
    Used for Optuna study names and MLflow metric names.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text
