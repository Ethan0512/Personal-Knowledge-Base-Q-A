from openai import OpenAI

# 替换成你刚创建的新 Key
API_KEY = "sk-903c5d9bf6364d8bad6ebb5b08b439ef"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

print("⏳ 正在调用 text-embedding-v4...")

response = client.embeddings.create(
    model="text-embedding-v4",
    input="你好，世界！"
)

print("✅ 调用成功！")
print(f"向量长度: {len(response.data[0].embedding)}")