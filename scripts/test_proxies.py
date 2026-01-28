import urllib.request
import json
import ssl
import sys
import time
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.utils.api_config import load_local_config, get_nested

CONFIG = load_local_config()

# Configuration (prefer local config, fallback to env)
ENDPOINTS = [
    {
        "name": "Local Proxy",
        "base_url": os.getenv("LOCAL_PROXY_BASE_URL") or get_nested(CONFIG, "local_proxy", "base_url"),
        "key": os.getenv("LOCAL_PROXY_KEY") or get_nested(CONFIG, "local_proxy", "api_key")
    },
    {
        "name": "Remote Proxy",
        "base_url": os.getenv("REMOTE_PROXY_BASE_URL") or get_nested(CONFIG, "remote_proxy", "base_url"),
        "key": os.getenv("REMOTE_PROXY_KEY") or get_nested(CONFIG, "remote_proxy", "api_key")
    }
]

def test_endpoint(endpoint):
    if not endpoint.get("base_url") or not endpoint.get("key"):
        return
    print(f"\n{'='*60}")
    print(f"Testing {endpoint['name']}")
    print(f"Base URL: {endpoint['base_url']}")
    print(f"{'='*60}")

    headers = {
        "Authorization": f"Bearer {endpoint['key']}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # 1. Test /models (GET)
    models_url = f"{endpoint['base_url'].rstrip('/')}/models"
    print(f"\n[1] Testing GET {models_url}...")
    
    available_models = []
    
    try:
        req = urllib.request.Request(models_url, headers=headers, method='GET')
        context = ssl._create_unverified_context()
        
        # Set a shorter timeout for local/remote connection checks
        with urllib.request.urlopen(req, context=context, timeout=10) as response:
            status = response.status
            print(f"    Status: {status}")
            
            if status == 200:
                body = response.read().decode('utf-8')
                try:
                    data = json.loads(body)
                    if 'data' in data:
                        models = data['data']
                        print(f"    Success! Found {len(models)} models.")
                        available_models = [m['id'] for m in models]
                        # Show first 5 models
                        for m in available_models[:5]:
                            print(f"    - {m}")
                    else:
                        print(f"    Warning: Response JSON missing 'data' field. Body preview: {body[:100]}")
                except json.JSONDecodeError:
                    print(f"    Error: Failed to decode JSON response. Body preview: {body[:100]}")
            else:
                print(f"    Failed with status code: {status}")
                
    except Exception as e:
        print(f"    Connection failed: {e}")

    # 2. Test /chat/completions (POST)
    if available_models:
        # Pick a model to test
        # Try to find a common model like gpt-3.5, llama, or just pick the first one
        test_model = available_models[0]
        for m in available_models:
            if 'gpt-3.5' in m or 'llama' in m.lower() or 'glm' in m.lower():
                test_model = m
                break
                
        print(f"\n[2] Testing Chat Completion with model: '{test_model}'...")
        chat_url = f"{endpoint['base_url'].rstrip('/')}/chat/completions"
        
        payload = {
            "model": test_model,
            "messages": [{"role": "user", "content": "Hello, are you working?"}],
            "max_tokens": 20
        }
        
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(chat_url, data=data, headers=headers, method='POST')
            context = ssl._create_unverified_context()
            
            with urllib.request.urlopen(req, context=context, timeout=20) as response:
                status = response.status
                print(f"    Status: {status}")
                
                if status == 200:
                    body = response.read().decode('utf-8')
                    try:
                        resp_json = json.loads(body)
                        content = resp_json['choices'][0]['message']['content']
                        print(f"    Success! Response: {content.strip()}")
                    except:
                        print(f"    Response body: {body[:200]}")
                else:
                    print(f"    Failed with status code: {status}")
                    
        except Exception as e:
            print(f"    Chat completion failed: {e}")
    else:
        print("\n[2] Skipping Chat Completion test (no models found or connection failed).")

if __name__ == "__main__":
    print("Starting Proxy Connection Tests...")
    available = [ep for ep in ENDPOINTS if ep.get("base_url") and ep.get("key")]
    if not available:
        print("No proxy endpoints configured. Set LOCAL_PROXY_BASE_URL/KEY or REMOTE_PROXY_BASE_URL/KEY.")
        sys.exit(1)
    for ep in available:
        test_endpoint(ep)
