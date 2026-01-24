#!/usr/bin/env python3
"""
Test model availability and basic stability for multiple API endpoints.

Reads endpoints from config/api.txt and optional --endpoint overrides that pull
keys from environment variables. Sends minimal prompts and reports success rate.
"""
import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests


DEFAULT_CONFIG = os.path.join("config", "api.txt")


def split_kv(line: str) -> str:
    if ":" in line:
        return line.split(":", 1)[1].strip()
    return ""


def normalize_name(name: str) -> str:
    return name.strip().lower()


def load_api_config(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []

    entries: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                if current:
                    entries.append(current)
                    current = {}
                continue

            lower = line.lower()
            if lower.startswith("key"):
                current["key"] = split_kv(line)
                continue
            if lower.startswith("url"):
                current["base_url"] = split_kv(line)
                continue

            if current:
                entries.append(current)
                current = {}
            current["name"] = line

    if current:
        entries.append(current)

    normalized: List[Dict[str, str]] = []
    for entry in entries:
        name = normalize_name(entry.get("name", ""))
        key = entry.get("key", "").strip()
        base_url = entry.get("base_url", "").strip()
        if name and key and base_url:
            normalized.append({"name": name, "key": key, "base_url": base_url})
    return normalized


def build_models_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if "/chat/completions" in url:
        url = url.split("/chat/completions", 1)[0]
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return f"{url}/models"


def build_openai_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if "/chat/completions" in url:
        return url
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return f"{url}/chat/completions"


def build_anthropic_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return f"{url}/messages"


def safe_snippet(text: str, limit: int = 120) -> str:
    trimmed = text.replace("\n", " ").replace("\r", " ").strip()
    return trimmed[:limit]


def fetch_models(endpoint: Dict[str, str], timeout: float) -> Tuple[List[str], Optional[str]]:
    url = build_models_url(endpoint["base_url"])
    headers = {"Authorization": f"Bearer {endpoint['key']}"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except Exception as exc:
        return [], f"request error: {exc}"

    if resp.status_code != 200:
        return [], f"HTTP {resp.status_code} {safe_snippet(resp.text)}"

    try:
        data = resp.json()
    except Exception:
        return [], "invalid json from /models"

    models: List[str] = []
    for item in data.get("data", []):
        model_id = item.get("id") or item.get("name")
        if model_id:
            models.append(model_id)
    return models, None


def extract_openai_text(data: Dict) -> str:
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def extract_anthropic_text(data: Dict) -> str:
    content = data.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            return str(first.get("text", "")).strip()
    if isinstance(content, str):
        return content.strip()
    return ""


def call_openai(endpoint: Dict[str, str], model: str, prompt: str, max_tokens: int,
                timeout: float) -> Tuple[bool, float, str]:
    url = build_openai_url(endpoint["base_url"])
    headers = {
        "Authorization": f"Bearer {endpoint['key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    start = time.perf_counter()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except Exception as exc:
        return False, 0.0, f"request error: {exc}"

    latency_ms = (time.perf_counter() - start) * 1000
    if resp.status_code != 200:
        return False, latency_ms, f"HTTP {resp.status_code} {safe_snippet(resp.text)}"
    try:
        data = resp.json()
    except Exception:
        return False, latency_ms, "invalid json from chat endpoint"
    text = extract_openai_text(data)
    if not text:
        return False, latency_ms, "empty response"
    return True, latency_ms, ""


def call_anthropic(endpoint: Dict[str, str], model: str, prompt: str, max_tokens: int,
                   timeout: float) -> Tuple[bool, float, str]:
    url = build_anthropic_url(endpoint["base_url"])
    headers = {
        "Authorization": f"Bearer {endpoint['key']}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    start = time.perf_counter()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except Exception as exc:
        return False, 0.0, f"request error: {exc}"

    latency_ms = (time.perf_counter() - start) * 1000
    if resp.status_code != 200:
        return False, latency_ms, f"HTTP {resp.status_code} {safe_snippet(resp.text)}"
    try:
        data = resp.json()
    except Exception:
        return False, latency_ms, "invalid json from anthropic endpoint"
    text = extract_anthropic_text(data)
    if not text:
        return False, latency_ms, "empty response"
    return True, latency_ms, ""


def filter_models(models: List[str], families: List[str],
                  include_regex: Optional[str], exclude_regex: Optional[str]) -> List[str]:
    filtered = models
    if families:
        family_patterns = [f.lower() for f in families]
        filtered = [m for m in filtered if any(p in m.lower() for p in family_patterns)]
    if include_regex:
        regex = re.compile(include_regex, re.IGNORECASE)
        filtered = [m for m in filtered if regex.search(m)]
    if exclude_regex:
        regex = re.compile(exclude_regex, re.IGNORECASE)
        filtered = [m for m in filtered if not regex.search(m)]
    return filtered


def parse_endpoint_arg(value: str) -> Dict[str, str]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) < 3:
        raise ValueError("endpoint format: name,base_url,key_env[,protocol]")
    name, base_url, key_env = parts[:3]
    protocol = parts[3] if len(parts) > 3 else "auto"
    key = os.environ.get(key_env, "")
    if not key:
        raise ValueError(f"missing env var: {key_env}")
    return {
        "name": normalize_name(name),
        "base_url": base_url,
        "key": key,
        "protocol": protocol,
        "key_source": f"env:{key_env}",
    }


def select_protocol(endpoint: Dict[str, str], default_protocol: str) -> str:
    protocol = endpoint.get("protocol", default_protocol)
    if protocol == "auto":
        if "/anthropic" in endpoint["base_url"]:
            return "anthropic"
        return "openai"
    return protocol


def run_trials(endpoint: Dict[str, str], model: str, protocol: str, prompt: str,
               max_tokens: int, timeout: float, trials: int, sleep_s: float) -> Dict:
    successes = 0
    latencies: List[float] = []
    error = ""
    for i in range(trials):
        if protocol == "anthropic":
            ok, latency, err = call_anthropic(endpoint, model, prompt, max_tokens, timeout)
        else:
            ok, latency, err = call_openai(endpoint, model, prompt, max_tokens, timeout)
        if ok:
            successes += 1
            latencies.append(latency)
        else:
            error = err
        if i < trials - 1 and sleep_s > 0:
            time.sleep(sleep_s)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "model": model,
        "success": successes,
        "trials": trials,
        "avg_latency_ms": round(avg_latency, 1),
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Test model availability for API endpoints.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to api.txt")
    parser.add_argument("--only", default="", help="Comma-separated endpoint names to test")
    parser.add_argument("--endpoint", action="append", default=[],
                        help="Extra endpoint: name,base_url,key_env[,protocol]")
    parser.add_argument("--family", default="", help="Comma-separated family filters")
    parser.add_argument("--match", default="", help="Regex to include models")
    parser.add_argument("--exclude", default="", help="Regex to exclude models")
    parser.add_argument("--models", default="", help="Comma-separated models to test")
    parser.add_argument("--max-models", type=int, default=3, help="Max models per endpoint")
    parser.add_argument("--trials", type=int, default=2, help="Trials per model")
    parser.add_argument("--timeout", type=float, default=30, help="Request timeout seconds")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between trials")
    parser.add_argument("--prompt", default="ping", help="Prompt for testing")
    parser.add_argument("--max-tokens", type=int, default=16, help="Max tokens for testing")
    parser.add_argument("--protocol", default="auto", choices=["auto", "openai", "anthropic"],
                        help="Default protocol for endpoints")
    parser.add_argument("--save", default="", help="Save JSON results to file")
    args = parser.parse_args()

    endpoints = load_api_config(args.config)
    for value in args.endpoint:
        endpoints.append(parse_endpoint_arg(value))

    only = [normalize_name(n) for n in args.only.split(",") if n.strip()]
    if only:
        endpoints = [e for e in endpoints if e["name"] in only]

    if not endpoints:
        print("No endpoints to test.")
        return 1

    families = [f.strip() for f in args.family.split(",") if f.strip()]
    include_regex = args.match.strip() or None
    exclude_regex = args.exclude.strip() or None
    manual_models = [m.strip() for m in args.models.split(",") if m.strip()]

    results = []
    for endpoint in endpoints:
        print(f"\n== {endpoint['name']} ==")
        protocol = select_protocol(endpoint, args.protocol)
        models = manual_models
        models_error = None
        if not models:
            models, models_error = fetch_models(endpoint, args.timeout)

        if models_error:
            print(f"models_error: {models_error}")

        if not models:
            print("no models to test (use --models to force).")
            continue

        models = filter_models(models, families, include_regex, exclude_regex)
        if args.max_models > 0:
            models = models[:args.max_models]

        if not models:
            print("no models after filters.")
            continue

        for model in models:
            stats = run_trials(
                endpoint=endpoint,
                model=model,
                protocol=protocol,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                trials=args.trials,
                sleep_s=args.sleep,
            )
            results.append({
                "endpoint": endpoint["name"],
                "protocol": protocol,
                **stats,
            })
            print(
                f"{model} -> {stats['success']}/{stats['trials']} "
                f"avg={stats['avg_latency_ms']}ms "
                f"{'OK' if stats['success'] == stats['trials'] else 'FAIL'}"
            )
            if stats["error"] and stats["success"] < stats["trials"]:
                print(f"  last_error: {stats['error']}")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        print(f"\nSaved results: {args.save}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
