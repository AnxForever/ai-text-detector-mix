"""
多 API 并行数据生成脚本
使用 10 个 API 端点并行生成，大幅提高速度
"""
import sys
import io
import os

# UTF-8 编码设置
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

import json
import time
import random
import requests
import pandas as pd
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parallel_generation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== API 端点配置 ====================
API_ENDPOINTS = [
    {
        "name": "wong",
        "models": ["deepseek-v3.2-chat", "qwen-max-latest", "claude-sonnet-4-5"]
    },
    {
        "name": "xiaoyo",
        "models": ["deepseek-v3.2", "Kimi-K2"]
    },
    {
        "name": "fovt",
        "models": ["deepseek-ai/DeepSeek-V3", "gpt-4.1-mini"]
    },
    {
        "name": "paolu",
        "models": ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"]
    },
    {
        "name": "b4u",
        "models": ["claude-4.5-sonnet", "claude-4-sonnet"]
    },
    {
        "name": "coderouter",
        "models": ["claude-haiku-4-5-20251001"]
    },
    {
        "name": "xiaodai",
        "models": ["anthropic/claude-3.7-sonnet", "qwen/qwen-2.5-72b-instruct"]
    },
    {
        "name": "liuge",
        "models": ["deepseek/deepseek-chat-v3-0324", "qwen/qwen-2.5-72b-instruct"]
    },
    {
        "name": "kfc",
        "models": ["cursor2-gpt-5"]
    },
    {
        "name": "bohe",
        "models": ["gpt-4.1-mini"]
    },
    {
        "name": "hybgzs_gemini",
        "models": ["gemini-2.5-flash"]
    },
    {
        "name": "hybgzs_gpt",
        "models": ["gpt-4.1-mini"]
    },
]

API_CONFIG_PATH = os.getenv("API_CONFIG_PATH", os.path.join("config", "api.txt"))

MODEL_FAMILY_PATTERNS = {
    "gpt": ["gpt", "openai"],
    "claude": ["claude"],
    "gemini": ["gemini"],
    "deepseek": ["deepseek"],
    "qwen": ["qwen", "kimi"]
}

DECODING_PRESETS = {
    "low": {
        "temperature": 0.2,
        "top_p": 0.8,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    },
    "medium": {
        "temperature": 0.7,
        "top_p": 0.9,
        "presence_penalty": 0.3,
        "frequency_penalty": 0.0
    },
    "high": {
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 0.6,
        "frequency_penalty": 0.0
    }
}

NAME_ALIASES = {
    "薄荷？": "bohe",
    "薄荷": "bohe"
}

ACTIVE_ENDPOINTS = []

# 全局配置
OUTPUT_DIR = "new_plan_datasets"
DEFAULT_PLAN_FILE = os.path.join(OUTPUT_DIR, "auto_generation_plan_v2_followup.json")
LEGACY_PLAN_FILE = os.path.join(OUTPUT_DIR, "auto_generation_plan.json")
PLAN_FILE = os.getenv("PLAN_FILE", DEFAULT_PLAN_FILE)
PARALLEL_OUTPUT_FILE = os.getenv("PARALLEL_OUTPUT_FILE", os.path.join(OUTPUT_DIR, "parallel_dataset.csv"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))  # 并行线程数
SAVE_INTERVAL = int(os.getenv("SAVE_INTERVAL", "50"))  # 每 50 条保存一次
TARGET_SAMPLES = int(os.getenv("TARGET_SAMPLES", "25000"))  # 目标生成数量
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "90"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
RATE_LIMIT_SLEEP = float(os.getenv("RATE_LIMIT_SLEEP", "6"))
ERROR_BACKOFF_SLEEP = float(os.getenv("ERROR_BACKOFF_SLEEP", "2"))

# 线程安全的结果收集
results_lock = threading.Lock()
results_queue = Queue()
save_lock = threading.Lock()
progress_counter = {"success": 0, "failed": 0}


def _split_kv(line: str) -> str:
    """Split key/value lines that may use ':' or '：'."""
    if ":" in line:
        return line.split(":", 1)[1].strip()
    if "：" in line:
        return line.split("：", 1)[1].strip()
    return ""


def _normalize_name(name: str) -> str:
    raw = name.strip()
    if not raw:
        return ""
    return NAME_ALIASES.get(raw, raw).strip().lower()


def load_api_config(config_path: str) -> dict:
    """Load api.txt into a name->config mapping."""
    if not os.path.exists(config_path):
        logger.warning(f"API config not found: {config_path}")
        return {}

    entries = []
    current = {}
    with open(config_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                if current:
                    entries.append(current)
                    current = {}
                continue

            lower = line.lower()
            if lower.startswith("key"):
                current["key"] = _split_kv(line)
                continue
            if lower.startswith("url"):
                current["base_url"] = _split_kv(line)
                continue

            if current:
                entries.append(current)
                current = {}
            current["name"] = line

    if current:
        entries.append(current)

    result = {}
    for entry in entries:
        name = _normalize_name(entry.get("name", ""))
        if not name:
            continue
        key = entry.get("key", "").strip()
        base_url = entry.get("base_url", "").strip()
        if key and base_url:
            result[name] = {"key": key, "base_url": base_url}

    return result


def build_api_url(base_url: str) -> str:
    """Build /chat/completions URL from a base URL or full endpoint."""
    url = base_url.rstrip("/")
    if "/chat/completions" in url:
        return url
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return f"{url}/chat/completions"


def build_family_models(models: list) -> dict:
    """Map model family -> model list."""
    family_models = {}
    for family, patterns in MODEL_FAMILY_PATTERNS.items():
        matched = []
        for model in models:
            model_lower = model.lower()
            if any(p in model_lower for p in patterns):
                matched.append(model)
        family_models[family] = matched
    return family_models


def resolve_api_endpoints() -> list:
    """Merge API_ENDPOINTS with api.txt credentials."""
    api_config = load_api_config(API_CONFIG_PATH)
    endpoints = []
    missing = []

    for endpoint in API_ENDPOINTS:
        name = endpoint["name"].lower()
        cfg = api_config.get(name)
        if not cfg:
            missing.append(endpoint["name"])
            continue

        merged = dict(endpoint)
        merged["key"] = cfg["key"]
        merged["base_url"] = cfg["base_url"]
        merged["url"] = build_api_url(cfg["base_url"])
        merged["family_models"] = build_family_models(endpoint["models"])
        endpoints.append(merged)

    if missing:
        logger.warning(f"Missing API config for: {', '.join(missing)}")

    return endpoints


def call_api(endpoint: dict, prompt: str, model: str, decoding: dict, max_tokens: int = 2000) -> tuple:
    """Call a single API endpoint."""
    headers = {
        "Authorization": f"Bearer {endpoint['key']}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": decoding.get("temperature", 0.7),
        "top_p": decoding.get("top_p", 0.9),
        "presence_penalty": decoding.get("presence_penalty", 0.0),
        "frequency_penalty": decoding.get("frequency_penalty", 0.0)
    }

    try:
        resp = requests.post(endpoint["url"], headers=headers, json=data, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            return content, endpoint['name'], model
        else:
            return None, endpoint['name'], f"HTTP {resp.status_code}"
    except Exception as e:
        return None, endpoint['name'], str(e)[:50]


def _select_endpoints_for_family(model_family: str, max_retries: int) -> list:
    candidates = []
    for endpoint in ACTIVE_ENDPOINTS:
        if endpoint["family_models"].get(model_family):
            candidates.append(endpoint)

    if not candidates:
        candidates = list(ACTIVE_ENDPOINTS)

    random.shuffle(candidates)
    return candidates[:min(max_retries, len(candidates))]


def _choose_model(endpoint: dict, model_family: str) -> str:
    family_models = endpoint["family_models"].get(model_family, [])
    if family_models:
        return random.choice(family_models)
    return random.choice(endpoint["models"])


def generate_with_retry(task_item: dict, max_retries: int = MAX_RETRIES) -> tuple:
    """Generate with retries and model-family routing."""
    prompt = task_item["prompt"]
    model_family = task_item.get("model_family", "mixed")
    decoding_profile = task_item.get("decoding_profile", "medium")
    decoding = DECODING_PRESETS.get(decoding_profile, DECODING_PRESETS["medium"])

    endpoints_to_try = _select_endpoints_for_family(model_family, max_retries)
    last_error = None

    for endpoint in endpoints_to_try:
        model = _choose_model(endpoint, model_family)
        result, api_name, model_used = call_api(endpoint, prompt, model, decoding)
        if result:
            return result, api_name, model_used
        last_error = f"{api_name}:{model_used}"
        if isinstance(model_used, str) and ("HTTP 429" in model_used or "Expecting value" in model_used):
            time.sleep(RATE_LIMIT_SLEEP)
        else:
            time.sleep(ERROR_BACKOFF_SLEEP)

    return None, "all_failed", last_error or "exhausted"


def process_task(task_item: dict, task_id: int) -> dict:
    """处理单个任务"""
    prompt = task_item["prompt"]

    result, api_name, model = generate_with_retry(task_item)

    if result:
        # 简单质量评估
        quality = 0.5
        if len(result) >= 300:
            quality += 0.2
        if '\n\n' in result or result.count('。') >= 3:
            quality += 0.1
        if "作为一个AI" not in result and "很抱歉" not in result:
            quality += 0.1
        if len(result) >= 500:
            quality += 0.1
        quality = min(quality, 1.0)

        return {
            "success": True,
            "text_id": f"parallel_{task_item['plan_id']}",
            "text_content": result,
            "source_api": api_name,
            "source_model": model,
            "attribute": task_item['attribute'],
            "topic": task_item['topic'],
            "genre": task_item['genre'],
            "role": task_item['role'],
            "style": task_item['style'],
            "constraint": task_item['constraint'],
            "prompt": prompt,
            "combination_quality": task_item['quality_score'],
            "generation_quality": quality,
            "length": len(result),
            "timestamp": datetime.now().isoformat(),
            "plan_id": task_item['plan_id']
        }
    else:
        return {
            "success": False,
            "plan_id": task_item['plan_id'],
            "error": model
        }


def save_results(records: list):
    """保存结果到 CSV"""
    if not records:
        return

    with save_lock:
        df = pd.DataFrame(records)

        # 移除内部字段
        save_columns = [c for c in df.columns if c not in ['success', 'error', 'plan_id']]
        df = df[save_columns]

        if os.path.exists(PARALLEL_OUTPUT_FILE):
            df.to_csv(PARALLEL_OUTPUT_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(PARALLEL_OUTPUT_FILE, index=False, encoding='utf-8-sig')

        logger.info(f"Saved {len(records)} records to {PARALLEL_OUTPUT_FILE}")


def update_plan(plan: list, completed_ids: set):
    """更新生成计划"""
    for item in plan:
        if item['plan_id'] in completed_ids:
            item['generated_parallel'] = True

    with open(PLAN_FILE, 'w', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    print("=" * 70, flush=True)
    print("        Multi-API Parallel Dataset Generation", flush=True)
    print("=" * 70, flush=True)
    print(f"Available API configs: {len(API_ENDPOINTS)}", flush=True)
    print(f"Max workers: {MAX_WORKERS}", flush=True)
    print(f"Target samples: {TARGET_SAMPLES}", flush=True)
    print("=" * 70, flush=True)

    # Resolve API endpoints from config
    global ACTIVE_ENDPOINTS
    ACTIVE_ENDPOINTS = resolve_api_endpoints()
    if not ACTIVE_ENDPOINTS:
        print(f"Error: No API endpoints available. Check {API_CONFIG_PATH}", flush=True)
        return
    print(f"Active endpoints: {len(ACTIVE_ENDPOINTS)}", flush=True)

    # 加载生成计划
    plan_file = PLAN_FILE
    if not os.path.exists(plan_file) and plan_file != LEGACY_PLAN_FILE and os.path.exists(LEGACY_PLAN_FILE):
        plan_file = LEGACY_PLAN_FILE

    if not os.path.exists(plan_file):
        print(f"Error: Plan file not found: {plan_file}", flush=True)
        return

    print(f"Plan file: {plan_file}", flush=True)

    with open(plan_file, 'r', encoding='utf-8') as f:
        generation_plan = json.load(f)

    print(f"Loaded plan with {len(generation_plan)} items", flush=True)

    # 筛选未生成的任务
    ungenerated = [p for p in generation_plan if not p.get('generated_parallel', False)]

    # 排除已经在其他模式下生成的
    for p in ungenerated:
        if any(p.get(f'generated_{m}', False) for m in ['deepseek', 'qwen', 'claude', 'gpt', 'gemini']):
            p['generated_parallel'] = True

    ungenerated = [p for p in generation_plan if not p.get('generated_parallel', False)]

    print(f"Ungenerated tasks: {len(ungenerated)}", flush=True)

    # 限制任务数量
    tasks_to_process = ungenerated[:TARGET_SAMPLES]
    print(f"Will process: {len(tasks_to_process)} tasks", flush=True)

    if not tasks_to_process:
        print("No tasks to process!", flush=True)
        return

    # 并行处理
    start_time = time.time()
    batch_results = []
    completed_ids = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for i, task in enumerate(tasks_to_process):
            future = executor.submit(process_task, task, i)
            futures[future] = (i, task['plan_id'])

        for future in as_completed(futures):
            idx, plan_id = futures[future]
            try:
                result = future.result()

                if result['success']:
                    batch_results.append(result)
                    completed_ids.add(plan_id)
                    progress_counter['success'] += 1

                    # 质量显示
                    q = result['generation_quality']
                    q_str = "Excellent" if q > 0.8 else "Good" if q > 0.6 else "OK"
                    print(f"[{progress_counter['success']}/{len(tasks_to_process)}] "
                          f"{q_str} q={q:.2f} len={result['length']} "
                          f"api={result['source_api']} attr={result['attribute']}", flush=True)
                else:
                    progress_counter['failed'] += 1
                    print(f"[FAIL] plan_id={plan_id} error={result.get('error', 'unknown')}", flush=True)

                # 定期保存
                if len(batch_results) >= SAVE_INTERVAL:
                    save_results(batch_results)
                    update_plan(generation_plan, completed_ids)
                    batch_results = []

                    # 打印统计
                    elapsed = time.time() - start_time
                    rate = progress_counter['success'] / (elapsed / 60) if elapsed > 0 else 0
                    print(f"\n=== Progress: {progress_counter['success']}/{len(tasks_to_process)} "
                          f"({progress_counter['failed']} failed) "
                          f"Rate: {rate:.1f}/min ===\n", flush=True)

            except Exception as e:
                progress_counter['failed'] += 1
                print(f"[ERROR] Task {idx}: {e}", flush=True)

    # 保存剩余结果
    if batch_results:
        save_results(batch_results)
        update_plan(generation_plan, completed_ids)

    # 最终统计
    elapsed = time.time() - start_time
    print("\n" + "=" * 70, flush=True)
    print("Generation Complete!", flush=True)
    print(f"Success: {progress_counter['success']}", flush=True)
    print(f"Failed: {progress_counter['failed']}", flush=True)
    print(f"Time: {elapsed/60:.1f} minutes", flush=True)
    print(f"Rate: {progress_counter['success']/(elapsed/60):.1f} samples/min", flush=True)
    print(f"Output: {PARALLEL_OUTPUT_FILE}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
