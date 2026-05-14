import requests

API_KEY = "sk-12257bf5efa14be5960ecc07fda2ded6"  # 换成刚复制的

url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
data = {
    "model": "deepseek-v3",  # 注意是 deepseek-v3，不是 deepseek-chat
    "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
    "stream": False
}

response = requests.post(url, headers=headers, json=data)
print("状态码:", response.status_code)
print("完整返回:", response.text)

if response.status_code == 200:
    result = response.json()
    print("模型回答:", result["choices"][0]["message"]["content"])
else:
    print("请求失败")