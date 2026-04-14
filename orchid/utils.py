"""Utility functions for post-experiment operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def update_metadata(data_dir: str | Path, **kwargs) -> dict:
    """Add or update fields in an experiment's metadata.yaml.

    Use this after an experiment to annotate results with additional
    information (notes, analysis results, quality flags, etc.).

    Parameters
    ----------
    data_dir : str or Path
        Path to the experiment directory (containing metadata.yaml).
    **kwargs
        Key-value pairs to add or overwrite in the metadata.

    Returns
    -------
    dict
        The full updated metadata dictionary.

    Examples
    --------
    >>> update_metadata("./data/0001", notes="good data", quality="A")
    >>> update_metadata(data_dir, T_mc=0.015, B_field="1T")
    """
    path = Path(data_dir) / "metadata.yaml"
    if path.exists():
        meta = yaml.safe_load(path.read_text()) or {}
    else:
        meta = {}

    meta.update(kwargs)
    path.write_text(yaml.safe_dump(meta, sort_keys=False))
    return meta


def read_metadata(data_dir: str | Path) -> dict:
    """Read an experiment's metadata.yaml.

    Parameters
    ----------
    data_dir : str or Path
        Path to the experiment directory.

    Returns
    -------
    dict
        The metadata dictionary.
    """
    path = Path(data_dir) / "metadata.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No metadata.yaml in {data_dir}")
    return yaml.safe_load(path.read_text()) or {}
