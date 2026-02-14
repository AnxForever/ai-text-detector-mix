#!/usr/bin/env python3
"""
P0 数据补全 - 批量调用 API 生成文本
读取 configs/p0_prompts.json，调用多个 API 生成 AI 文本，输出到 datasets/p0_generated/

用法：
    python scripts/generation/gen_p0_texts.py
    python scripts/generation/gen_p0_texts.py --dry-run          # 只打印不调用
    python scripts/generation/gen_p0_texts.py --start-from 100   # 从第100条开始（断点续传）
    python scripts/generation/gen_p0_texts.py --limit 50         # 只生成50条
"""

import json
import time
import random
import argparse
import csv
import os
from pathlib import Path
from datetime import datetime

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ============================================================
# API 配置
# ============================================================

API_ENDPOINTS = {
    "x666": {
        "base_url": "https://x666.me/v1",
        "api_key": os.getenv("P0_X666_API_KEY", ""),
        "models": [
            "gemini-2.5-flash",
            "gemini-3-flash-preview",
        ],
        "rpm_limit": 20,
        "verify_ssl": True,
    },
    "yetong": {
        "base_url": "http://manyrouter.chaosyn.com/v1",
        "api_key": os.getenv("P0_YETONG_API_KEY", ""),
        "models": [
            "claude-sonnet-4-5-20250929",
        ],
        "rpm_limit": 10,
        "verify_ssl": True,
    },
    # any_manyrouter: DISABLED - HTTP 400
    # yingjiang: DISABLED - 额度用尽
    # cifang: DISABLED - 令牌不可用
    "heiyubai": {
        "base_url": "https://ai.hybgzs.com/v1",
        "api_key": os.getenv("P0_HEIYUBAI_API_KEY", ""),
        "models": [
            "gemini-2.5-flash",
            "gemini-3-flash-preview",
        ],
        "rpm_limit": 10,
        "verify_ssl": False,
        "no_proxy": True,
    },
}


def build_model_to_endpoint() -> list[tuple[str, str, dict]]:
    """Build (model_name, endpoint_name, endpoint_config) list."""
    pairs: list[tuple[str, str, dict]] = []
    for ep_name, ep_config in API_ENDPOINTS.items():
        api_key = ep_config.get("api_key", "").strip()
        if not api_key:
            print(f"[WARN] skip endpoint without api key: {ep_name}", flush=True)
            continue
        for model in ep_config["models"]:
            pairs.append((model, ep_name, ep_config))
    return pairs


# ============================================================
# API 调用
# ============================================================

def call_api(base_url: str, api_key: str, model: str, prompt: str,
             max_tokens: int = 800, temperature: float = 0.9,
             endpoint_name: str = "", max_retries: int = 3,
             verify_ssl: bool = True, no_proxy: bool = False) -> str | None:
    """调用 OpenAI 兼容 API，带重试，返回生成文本或 None"""
    for attempt in range(max_retries):
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            # x666 对 max_tokens 有 bug，不传
            if endpoint_name != "x666":
                payload["max_tokens"] = max_tokens

            kwargs = dict(
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=90,
                verify=verify_ssl,
            )
            if no_proxy:
                kwargs["proxies"] = {"http": "", "https": ""}

            r = requests.post(
                f"{base_url}/chat/completions",
                **kwargs,
            )
            if r.status_code == 200:
                data = r.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content and content.strip():
                    return content.strip()
                # body 可能包含错误信息
                msg = data.get("msg", "")
                if msg:
                    print(f"    [API-MSG] {msg[:100]}", flush=True)
                    return None
            elif r.status_code in (429, 502, 503, 524, 525):
                wait = (attempt + 1) * 3
                print(f"    [HTTP {r.status_code}] retry {attempt+1}/{max_retries}, wait {wait}s", flush=True)
                time.sleep(wait)
                continue
            else:
                print(f"    [HTTP {r.status_code}] {r.text[:120]}", flush=True)
                return None
        except requests.exceptions.Timeout:
            wait = (attempt + 1) * 3
            print(f"    [TIMEOUT] retry {attempt+1}/{max_retries}, wait {wait}s", flush=True)
            time.sleep(wait)
            continue
        except Exception as e:
            print(f"    [ERROR] {model}: {e}", flush=True)
            return None
    return None


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="P0 批量文本生成")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "configs" / "p0_prompts.json"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "datasets" / "p0_generated"))
    parser.add_argument("--dry-run", action="store_true", help="只打印不调用 API")
    parser.add_argument("--start-from", type=int, default=0, help="从第N条开始（断点续传）")
    parser.add_argument("--limit", type=int, default=0, help="最多生成N条（0=全部）")
    parser.add_argument("--delay", type=float, default=2.0, help="请求间隔秒数")
    args = parser.parse_args()

    # 读取 prompt
    with open(args.input, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    print(f"=== P0 批量文本生成 ===", flush=True)
    print(f"总 prompt: {len(prompts)}", flush=True)
    print(f"起始位置: {args.start_from}", flush=True)
    print(f"请求间隔: {args.delay}s", flush=True)

    # 准备输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建模型路由
    model_pool = build_model_to_endpoint()
    print(f"可用模型: {len(model_pool)} 个", flush=True)
    for model, ep, _ in model_pool:
        print(f"  {model} @ {ep}", flush=True)

    # 输出文件（JSONL 追加模式，支持断点续传，换行安全）
    output_jsonl = output_dir / "p0_generated.jsonl"

    # 已生成的 ID（断点续传用）
    done_ids = set()
    if output_jsonl.exists():
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        done_ids.add(int(row.get("prompt_id", 0)))
                    except (json.JSONDecodeError, ValueError):
                        pass
        print(f"已完成: {len(done_ids)} 条", flush=True)

    outfile = open(output_jsonl, "a", encoding="utf-8")

    # 统计
    success = 0
    fail = 0
    skipped = 0

    # 遍历 prompt
    subset = prompts[args.start_from:]
    if args.limit > 0:
        subset = subset[:args.limit]

    for i, item in enumerate(subset):
        pid = item["id"]
        if pid in done_ids:
            skipped += 1
            continue

        # 随机选一个模型
        model, ep_name, ep_config = random.choice(model_pool)

        print(f"[{i+1}/{len(subset)}] id={pid} cat={item['category']} "
              f"tpl={item['template']} -> {model}@{ep_name}", flush=True)

        if args.dry_run:
            print(f"    [DRY-RUN] prompt: {item['prompt'][:80]}...", flush=True)
            continue

        text = call_api(
            ep_config["base_url"],
            ep_config["api_key"],
            model,
            item["prompt"],
            endpoint_name=ep_name,
            verify_ssl=ep_config.get("verify_ssl", True),
            no_proxy=ep_config.get("no_proxy", False),
        )

        if text and len(text) > 20:
            record = {
                "prompt_id": pid,
                "category": item["category"],
                "template": item["template"],
                "topic": item["topic"][:100],
                "text": text,
                "label": 1,
                "source": f"p0_{item['category']}_{model}",
                "model": model,
                "endpoint": ep_name,
                "timestamp": datetime.now().isoformat(),
            }
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            outfile.flush()
            success += 1
            print(f"    [OK] {len(text)} chars", flush=True)
        else:
            fail += 1
            print(f"    [FAIL] empty or too short", flush=True)

        # 限速
        time.sleep(args.delay)

    outfile.close()

    print(f"\n=== 完成 ===", flush=True)
    print(f"成功: {success}", flush=True)
    print(f"失败: {fail}", flush=True)
    print(f"跳过(已完成): {skipped}", flush=True)
    print(f"输出: {output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
