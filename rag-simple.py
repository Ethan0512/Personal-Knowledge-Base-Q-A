import dashscope
from dashscope import TextEmbedding
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import numpy as np

YOUR_API_KEY = "sk-903c5d9bf6364d8bad6ebb5b08b439ef"
dashscope.api_key = YOUR_API_KEY

os.environ["OPENAI_API_KEY"] = YOUR_API_KEY
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ---------- 1. 加载和切分文档 ----------
loader = TextLoader("test.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
texts = [doc.page_content for doc in docs]
print(f"✅ 文档已切分为 {len(texts)} 个片段")

print("⏳ 正在生成向量...")
embeddings_list = []
for text in texts:
    resp = TextEmbedding.call(
        model="text-embedding-v4",  # 和 test_embedding.py 里的模型保持一致
        input=text
    )
    if resp.status_code == 200:
        embeddings_list.append(resp.output["embeddings"][0]["embedding"])
    else:
        print(f"❌ Embedding 失败: {resp}")
        exit(1)  # 如果失败就停止，方便排查
print(f"✅ 成功生成 {len(embeddings_list)} 个向量，维度 {len(embeddings_list[0])}")


# ---------- 3. 极简检索函数 (不用向量数据库，直接计算相似度) ----------
def simple_retrieve(query, embeddings_list, texts, top_k=2):
    # 获取问题的向量
    resp = TextEmbedding.call(model="text-embedding-v4", input=query)
    if resp.status_code != 200:
        return []
    query_vec = resp.output["embeddings"][0]["embedding"]

    # 计算余弦相似度
    similarities = []
    for doc_vec in embeddings_list:
        dot = np.dot(query_vec, doc_vec)
        norm_q = np.linalg.norm(query_vec)
        norm_d = np.linalg.norm(doc_vec)
        if norm_q == 0 or norm_d == 0:
            sim = 0
        else:
            sim = dot / (norm_q * norm_d)
        similarities.append(sim)

    # 取 top_k 个最相似的索引
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [texts[i] for i in top_indices]


# ---------- 4. 初始化大模型和提示词 ----------
prompt = ChatPromptTemplate.from_template("""
你是一个知识助手。请根据以下参考资料回答用户的问题。

参考资料：
{context}

用户问题：{question}

请基于参考资料回答，如果没有相关信息，请如实告知。
""")
llm = ChatOpenAI(model="deepseek-v3", temperature=0)


def ask_question(question: str) -> str:
    # 检索
    docs = simple_retrieve(question, embeddings_list, texts)
    context = "\n\n".join(docs)
    # 生成
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    return response.content


# ---------- 5. 测试 ----------
print("\n" + "=" * 50)
q1 = "你好？"
print(f"📝 问题: {q1}")
a1 = ask_question(q1)
print(f"💡 答案: {a1}")

print("\n" + "=" * 50)
q2 = "机器学习有哪些类型？"
print(f"📝 问题: {q2}")
a2 = ask_question(q2)
print(f"💡 答案: {a2}")
