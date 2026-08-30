"""Application settings: environment variables plus ``configs/config.yaml``.

YAML is the source of default alpha, power, mSPRT mixture variance, and
Bayesian prior hyperparameters used by :mod:`abtestlab.stats.core` (the API
backend). The typed library in :mod:`abtestlab.functional` does not read this
file.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Process-level settings loaded from the environment and optional ``.env``.

    Attributes:
        config_path: Path to the YAML defaults file for the API stats stack.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    config_path: Path = REPO_ROOT / "configs" / "config.yaml"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached process settings.

    Returns:
        A :class:`Settings` instance.
    """
    return Settings()


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a YAML mapping from ``path``.

    Args:
        path: Filesystem path to a YAML document.

    Returns:
        The parsed mapping.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is empty or is not a YAML mapping.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(loaded).__name__}")
    return loaded


@functools.lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Return the cached YAML config for the API stats stack.

    Returns:
        Config mapping from :attr:`Settings.config_path`.

    Raises:
        FileNotFoundError: If the configured path is missing.
        ValueError: If the file is not a YAML mapping.
    """
    return load_config(get_settings().config_path)


def resolve_path(relative: str | Path) -> Path:
    """Resolve ``relative`` against the repository root unless it is absolute.

    Args:
        relative: Path or path-like string.

    Returns:
        An absolute :class:`~pathlib.Path`.
    """
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p
