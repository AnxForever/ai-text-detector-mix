#!/usr/bin/env python3
"""测试V0 API连接"""
import requests
import json

V0_API_KEY = "v1:v0sjstHSQIP9l22wEx8gbkjr:yz4uuu0EZoGhKtXM2uy5wPdt"
V0_API_BASE = "https://api.v0.dev/v1"

def test_v0_api():
    """测试V0 API连接"""
    print("=" * 60)
    print("V0 API 连接测试")
    print("=" * 60)
    print(f"API端点: {V0_API_BASE}")
    print(f"API密钥: {V0_API_KEY[:20]}...")
    print()

    try:
        response = requests.post(
            f"{V0_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {V0_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "你好，请回复：V0 API连接成功！"}],
                "max_tokens": 50
            },
            timeout=15
        )

        print(f"状态码: {response.status_code}")
        print()

        if response.status_code == 200:
            result = response.json()
            print("✓ V0 API连接成功！")
            print(f"使用的模型: {result.get('model', 'unknown')}")
            print(f"回复内容: {result['choices'][0]['message']['content']}")
            print()
            print("=" * 60)
            print("测试通过！V0 API配置正确。")
            print("=" * 60)
            return True
        else:
            print(f"✗ V0 API连接失败")
            print(f"错误信息: {response.text}")
            return False

    except Exception as e:
        print(f"✗ 连接出错: {e}")
        return False

if __name__ == "__main__":
    success = test_v0_api()
    exit(0 if success else 1)