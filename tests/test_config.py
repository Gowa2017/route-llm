"""Tests for config loading."""

from pathlib import Path

import pytest
import tomllib

from llmrouter.config import _apply_env_overrides, load_config
from llmrouter.models import AppConfig

SAMPLE_TOML = """
[routing]
default_provider = "openai.deepseek"

[[routing.rules]]
name = "daytime"
time_range = { start = "08:00", end = "23:00" }
provider = "openai.deepseek"
model = "deepseek-chat"
priority = 10

[providers]
[providers.openai]
[providers.openai.deepseek]
api_key = "sk-test"
base_url = "https://api.deepseek.com"

[providers.openai.openai_vendor]
api_key = "sk-oa-test"
base_url = "https://api.openai.com"

[providers.anthropic]
[providers.anthropic.default]
api_key = "sk-ant-test"
base_url = "https://api.anthropic.com"
"""


def test_load_config_from_path(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(SAMPLE_TOML)
    cfg = load_config(cfg_path)

    assert isinstance(cfg, AppConfig)
    assert cfg.default_provider == "openai.deepseek"
    assert len(cfg.rules) == 1
    assert cfg.rules[0].name == "daytime"
    assert cfg.rules[0].provider == "openai.deepseek"
    assert cfg.rules[0].model == "deepseek-chat"
    assert len(cfg.providers) == 2
    assert "deepseek" in cfg.providers["openai"]
    assert "openai_vendor" in cfg.providers["openai"]
    assert cfg.providers["openai"]["deepseek"].api_key == "sk-test"
    assert cfg.providers["anthropic"]["default"].base_url == "https://api.anthropic.com"


def test_env_override_api_key(monkeypatch):
    raw = {
        "routing": {"default_provider": "openai.deepseek"},
        "providers": {
            "openai": {
                "deepseek": {"api_key": "sk-old", "base_url": "https://api.deepseek.com"}
            }
        },
    }
    monkeypatch.setenv("LLMROUTER_PROVIDER_OPENAI_DEEPSEEK_API_KEY", "sk-new")
    result = _apply_env_overrides(raw)
    assert result["providers"]["openai"]["deepseek"]["api_key"] == "sk-new"


def test_env_override_base_url(monkeypatch):
    raw = {
        "routing": {"default_provider": "openai.deepseek"},
        "providers": {
            "openai": {
                "deepseek": {"api_key": "sk-test", "base_url": "https://old.example.com"}
            }
        },
    }
    monkeypatch.setenv("LLMROUTER_PROVIDER_OPENAI_DEEPSEEK_BASE_URL", "https://new.example.com")
    result = _apply_env_overrides(raw)
    assert result["providers"]["openai"]["deepseek"]["base_url"] == "https://new.example.com"


def test_env_override_unknown_provider_noop(monkeypatch):
    raw = {
        "providers": {
            "openai": {
                "deepseek": {"api_key": "sk-test", "base_url": "https://x.com"}
            }
        },
    }
    monkeypatch.setenv("LLMROUTER_PROVIDER_OPENAI_UNKNOWN_API_KEY", "sk-xxx")
    result = _apply_env_overrides(raw)
    assert result["providers"]["openai"]["deepseek"]["api_key"] == "sk-test"


def test_env_override_unknown_type_noop(monkeypatch):
    raw = {
        "providers": {
            "openai": {
                "deepseek": {"api_key": "sk-test", "base_url": "https://x.com"}
            }
        },
    }
    monkeypatch.setenv("LLMROUTER_PROVIDER_UNKNOWN_DEEPSEEK_API_KEY", "sk-xxx")
    result = _apply_env_overrides(raw)
    assert result["providers"]["openai"]["deepseek"]["api_key"] == "sk-test"


def test_routing_rule_without_time_range():
    toml = """
[[routing.rules]]
name = "always"
provider = "openai.deepseek"
model = "deepseek-chat"
priority = 5
"""
    raw = tomllib.loads(toml)
    raw.update(raw.pop("routing", {}))
    cfg = AppConfig(**raw)
    assert len(cfg.rules) == 1
    assert cfg.rules[0].time_range is None
