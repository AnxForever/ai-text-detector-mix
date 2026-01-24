#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""娴嬭瘯鏂版柟妗圓PI杩炴帴"""

import requests

API_KEY = "sk-***"
API_BASE = "https://wzw.pp.ua/v1"

MODELS = {
    "deepseek": "deepseek-v3.2-chat",
    "claude": "claude-sonnet-4-5",
    "qwen": "qwen-max-latest",
    "glm": "GLM-4.7"
}

def test_model(name, model_name):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": "璇峰洖澶?娴嬭瘯鎴愬姛'"}],
        "max_tokens": 20
    }
    try:
        resp = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=data, timeout=30)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            print(f"鉁?{name.upper():10s} - {content[:30]}")
            return True
        else:
            print(f"鉂?{name.upper():10s} - HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"鉂?{name.upper():10s} - {e}")
        return False

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 60)
    print("鏂版柟妗圓PI杩炴帴娴嬭瘯")
    print("=" * 60)
    results = []
    for name, model in MODELS.items():
        results.append((name, test_model(name, model)))
    print("=" * 60)
    success = sum(1 for _, r in results if r)
    print(f"鍙敤妯″瀷: {success}/{len(MODELS)}")
    if success == len(MODELS):
        print("鎵€鏈夋ā鍨嬪彲鐢紝鍙互寮€濮嬫柊鏂规鐢熸垚")
    else:
        print("閮ㄥ垎妯″瀷涓嶅彲鐢?)

