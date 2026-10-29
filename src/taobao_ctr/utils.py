"""Small cross-cutting helpers used by more than one package (notebook section 2)."""
import re


def slug(s: str) -> str:
    """A filesystem/study-name-safe identifier, e.g. for prediction files and Optuna study names."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
