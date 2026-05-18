"""TOML config loading and validation."""

import os
import tomllib
from pathlib import Path

from llmrouter.models import AppConfig


_CONFIG_ENV_VAR = "LLMROUTER_CONFIG"
_DEFAULT_PATHS = [
    Path("config.toml"),
    Path.home() / ".config" / "llmrouter" / "config.toml",
    Path("/etc/llmrouter/config.toml"),
]


def _resolve_config_path() -> Path:
    """Resolve config file path from env var or default locations."""
    if env_path := os.environ.get(_CONFIG_ENV_VAR):
        return Path(env_path)

    for p in _DEFAULT_PATHS:
        if p.exists():
            return p

    msg = (
        f"No config found. Set {_CONFIG_ENV_VAR} or place config.toml in: "
        + ", ".join(str(p) for p in _DEFAULT_PATHS)
    )
    raise FileNotFoundError(msg)


def _apply_env_overrides(raw: dict) -> dict:
    """Override provider api_key / base_url from env vars.

    LLMROUTER_PROVIDER_<TYPE>_<NAME>_API_KEY -> providers.<type>.<name>.api_key
    LLMROUTER_PROVIDER_<TYPE>_<NAME>_BASE_URL -> providers.<type>.<name>.base_url
    """
    providers = raw.get("providers", {})
    for proto_type, vendors in providers.items():
        for vendor_name in vendors:
            env_api = f"LLMROUTER_PROVIDER_{proto_type.upper()}_{vendor_name.upper()}_API_KEY"
            env_url = f"LLMROUTER_PROVIDER_{proto_type.upper()}_{vendor_name.upper()}_BASE_URL"
            if env_api in os.environ:
                providers[proto_type][vendor_name]["api_key"] = os.environ[env_api]
            if env_url in os.environ:
                providers[proto_type][vendor_name]["base_url"] = os.environ[env_url]
    return raw


def load_config(path: Path | None = None) -> AppConfig:
    """Load and validate TOML config from *path* or auto-discover."""
    config_path = path or _resolve_config_path()
    raw = tomllib.loads(config_path.read_text())
    raw = _apply_env_overrides(raw)

    # Flatten [routing] section into top-level keys
    if "routing" in raw:
        raw.update(raw.pop("routing"))

    return AppConfig(**raw)
