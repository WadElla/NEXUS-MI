"""Portable data/output path handling.

NEXUS-MI runtime paths are resolved from user configuration or repository-relative defaults.
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Resolve the checkout root for editable use, with safe installed fallback."""
    explicit = os.environ.get("NEXUS_MI_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "nexus_mi").is_dir():
            return parent

    # A non-editable site-packages install does not contain the Git checkout.
    # In that case default local data/outputs to the user's current project
    # directory rather than writing beside site-packages.
    return Path.cwd().resolve()


def data_root() -> Path:
    return Path(os.environ.get("NEXUS_MI_DATA_DIR", repo_root() / "data")).expanduser().resolve()


def output_root() -> Path:
    return Path(os.environ.get("NEXUS_MI_OUTPUT_DIR", repo_root() / "outputs")).expanduser().resolve()


def dataset_dir(dataset: str) -> Path:
    name = dataset.lower()
    aliases = {"bci42a": "bciciv2a", "korea": "openbmi", "bciciv-2a": "bciciv2a"}
    return data_root() / aliases.get(name, name)
