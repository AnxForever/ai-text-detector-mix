import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repo root on sys.path for local imports if needed
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def load_local_config() -> Dict[str, Any]:
    """Load local API config from config/api.local.json if present."""
    config_path = Path(__file__).resolve().parents[2] / "config" / "api.local.json"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_nested(config: Dict[str, Any], *keys: str, default: Optional[str] = None) -> Optional[str]:
    """Safely get nested config values."""
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    if cur is None:
        return default
    return cur


def get_proxy_config(proxy_name: str, env_prefix: str, default_base_url: Optional[str] = None) -> Dict[str, str]:
    """Return proxy config (base_url/key) from env or local config."""
    config = load_local_config()
    base_url = os.getenv(f"{env_prefix}_BASE_URL") or get_nested(config, proxy_name, "base_url") or default_base_url
    api_key = os.getenv(f"{env_prefix}_KEY") or get_nested(config, proxy_name, "api_key")
    if not base_url or not api_key:
        raise RuntimeError(
            f"Missing proxy config for {proxy_name}. Set {env_prefix}_BASE_URL/{env_prefix}_KEY "
            "or config/api.local.json."
        )
    return {"url": base_url, "key": api_key}
