import urllib.request
import json
import ssl
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.utils.api_config import load_local_config, get_nested

CONFIG = load_local_config()
API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("NVAPI_KEY") or get_nested(CONFIG, "nvidia", "api_key")
BASE_URL = os.getenv("NVIDIA_BASE_URL") or get_nested(CONFIG, "nvidia", "base_url") or "https://integrate.api.nvidia.com/v1"
URL = f"{BASE_URL.rstrip('/')}/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0"
}

payload = {
    "model": "moonshotai/kimi-k2.5",
    "messages": [
        {"role": "user", "content": "你好，请介绍一下你自己。"}
    ],
    "temperature": 0.5,
    "max_tokens": 100
}


def test_get_models_list():
    if not API_KEY:
        print("Missing API key. Set NVIDIA_API_KEY (or NVAPI_KEY) in your environment.")
        return []
    print(f"\nFetching model list from {URL.replace('/chat/completions', '/models')}...")
    try:
        req = urllib.request.Request(URL.replace('/chat/completions', '/models'), headers=headers, method='GET')
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=context, timeout=30) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                return [m['id'] for m in data.get('data', [])]
            return []
    except Exception as e:
        print(f"GET models failed: {e}")
        return []


def test_chat_simple(model_name):
    if not API_KEY:
        print("Missing API key. Set NVIDIA_API_KEY (or NVAPI_KEY) in your environment.")
        return False
    print(f"Testing {model_name}...", end=" ", flush=True)
    p = payload.copy()
    p['model'] = model_name
    p['max_tokens'] = 10

    try:
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(URL, data=data, headers=headers, method='POST')
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=context, timeout=20) as response:
            if response.status == 200:
                print("OK")
                return True
            print(f"Failed ({response.status})")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    all_models = test_get_models_list()
    if not all_models:
        print("Could not retrieve model list.")
        sys.exit(1)

    print(f"Total models available: {len(all_models)}")

    families = {
        "Llama 3.1": ["meta/llama-3.1-70b-instruct", "meta/llama-3.1-8b-instruct", "meta/llama-3.1-405b-instruct"],
        "Qwen": ["qwen/qwen2.5-72b-instruct", "qwen/qwen2.5-7b-instruct", "qwen/qwen-2-7b-instruct"],
        "DeepSeek": ["deepseek-ai/deepseek-r1", "deepseek-ai/deepseek-v3", "deepseek-ai/deepseek-r1-distill-llama-70b"],
        "Mistral": ["mistralai/mistral-large-2-instruct", "mistralai/mixtral-8x22b-instruct-v0.1", "mistralai/mistral-nemo-12b-instruct"],
        "Gemma": ["google/gemma-2-27b-it", "google/gemma-2-9b-it"],
        "Phi": ["microsoft/phi-3.5-mini-instruct", "microsoft/phi-4"],
        "GLM": ["z-ai/glm4.7", "thudm/glm-4-9b-chat"],
        "Yi": ["01-ai/yi-large", "01-ai/yi-1.5-34b-chat"]
    }

    stable_models = []

    print("\nStarting Stability Check...")
    print("-" * 50)

    for family, candidates in families.items():
        print(f"\nChecking {family} family:")
        available_candidates = [c for c in candidates if c in all_models]

        if not available_candidates:
            keyword = family.lower().split()[0]
            available_candidates = [m for m in all_models if keyword in m.lower()][:3]

        for model in available_candidates:
            if test_chat_simple(model):
                stable_models.append(model)

    print("-" * 50)
    print("\nSummary of Stable Models:")
    for m in stable_models:
        print(f"- {m}")
