"""Config loading error paths."""

from __future__ import annotations

import pytest

from abtestlab.settings import get_config, get_settings, load_config


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_rejects_non_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(path)


def test_get_config_missing_path(monkeypatch, tmp_path):
    class FakeSettings:
        config_path = tmp_path / "absent.yaml"

    get_config.cache_clear()
    monkeypatch.setattr("abtestlab.settings.get_settings", lambda: FakeSettings())
    with pytest.raises(FileNotFoundError):
        get_config()
    get_config.cache_clear()


def test_get_config_repo_defaults_have_expected_keys():
    get_settings.cache_clear()
    get_config.cache_clear()
    cfg = get_config()
    assert "defaults" in cfg and "sequential" in cfg and "bayes" in cfg
    assert 0 < cfg["defaults"]["alpha"] < 1
