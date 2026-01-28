import requests
import json
import sys
import os
import urllib3

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.utils.api_config import load_local_config, get_nested

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration (prefer local config, fallback to env)
CONFIG = load_local_config()
API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("NVAPI_KEY") or get_nested(CONFIG, "nvidia", "api_key")
BASE_URL = os.getenv("NVIDIA_BASE_URL") or get_nested(CONFIG, "nvidia", "base_url") or "https://integrate.api.nvidia.com/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

def list_models():
    if not API_KEY:
        print("Missing API key. Set NVIDIA_API_KEY (or NVAPI_KEY) in your environment.")
        return None
    print(f"Checking available models at {BASE_URL}/models ...")
    try:
        response = requests.get(f"{BASE_URL}/models", headers=headers, verify=False, timeout=30)
        if response.status_code == 200:
            models_data = response.json()
            data = models_data.get('data', [])
            print(f"Found {len(data)} models.")

            # Filter for GLM models
            kimi_models = [m['id'] for m in data if 'kimi' in m['id'].lower()]

            if kimi_models:
                print("Found Kimi related models:")
                for m in kimi_models:
                    print(f" - {m}")

                # Prioritize moonshotai/kimi-k2.5
                target = "moonshotai/kimi-k2.5"
                if target in kimi_models:
                    return target
                return kimi_models[0] # Return the first one as a candidate

            # Filter for GLM models (legacy)
            glm_models = [m['id'] for m in data if 'glm' in m['id'].lower()]

            if glm_models:
                print("Found GLM related models:")
                for m in glm_models:
                    print(f" - {m}")
                return glm_models[0] # Return the first one as a candidate
            else:
                print("No models with 'glm' in the name found. Listing first 5 models:")
                for m in data[:5]:
                    print(f" - {m['id']}")
                return None
        else:
            print(f"Error listing models: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Exception listing models: {e}")
        return None

def test_chat(model_id):
    if not API_KEY:
        print("Missing API key. Set NVIDIA_API_KEY (or NVAPI_KEY) in your environment.")
        return
    print(f"\nTesting chat completion with model: {model_id}")
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "你好，请介绍一下你自己。"}
        ],
        "temperature": 0.5,
        "max_tokens": 100,
        "stream": False
    }

    headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

    try:
        print("Sending POST request...")
        session = requests.Session()
        session.verify = False
        response = session.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60)
        print(f"Response received. Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("Response success!")
            print("-" * 30)
            if 'choices' in result and len(result['choices']) > 0:
                print("Full response content:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                print("Message content:", result['choices'][0]['message'].get('content'))
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            print("-" * 30)
        else:
            print(f"Error in chat completion: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Exception in chat completion: {e}")

if __name__ == "__main__":
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)

    # 1. Try to list models to find the correct ID
    target_model = list_models()
    # target_model = "moonshotai/kimi-k2.5" # Hardcoded based on user request

    # 2. If user specified 'glm4.7' and we didn't find a better match, we might try to guess or use what was found.
    # The user asked for "glm4.7". Let's check if we found a match.

    if target_model:
        print(f"\nProceeding with auto-detected model: {target_model}")
        test_chat(target_model)
    else:
        # Fallback: try 'thudm/glm-4-9b-chat' or just 'glm-4' as common defaults if list failed or returned nothing
        fallback_model = "thudm/glm-4-9b-chat"
        print(f"\nCould not detect specific GLM model from list. Trying fallback: {fallback_model}")
        test_chat(fallback_model)
