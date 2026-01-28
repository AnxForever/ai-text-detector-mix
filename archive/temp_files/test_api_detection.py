import requests
import json

# 测试API检测
text = "随着人工智能技术的快速发展，AI文本检测变得越来越重要。"

response = requests.post(
    "http://localhost:8000/api/detect",
    json={"text": text}
)

print("状态码:", response.status_code)
print("响应:", json.dumps(response.json(), indent=2, ensure_ascii=False))
