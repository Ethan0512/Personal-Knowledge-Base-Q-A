from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # 使用最新的包
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

# ========= 使用你成功测试过的 API Key =========
YOUR_API_KEY = "sk-12257bf5efa14be5960ecc07fda2ded6"
# =============================================

# 配置环境变量
os.environ["OPENAI_API_KEY"] = YOUR_API_KEY
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 1. 加载文档
loader = TextLoader("test.txt", encoding="utf-8")
documents = loader.load()
print(f"✅ 加载了 {len(documents)} 个文档")

# 2. 切分文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
print(f"✅ 切分成了 {len(docs)} 个文本块")

# 3. 创建向量数据库
print("⏳ 正在创建向量数据库（调用阿里云 Embedding API）...")
embeddings = OpenAIEmbeddings(model="text-embedding-v3")

# 提取文本内容
texts = [doc.page_content for doc in docs]
metadatas = [doc.metadata for doc in docs]
vectorstore = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
print("✅ 向量数据库创建完成")

# 4. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 5. 创建提示词模板
prompt = ChatPromptTemplate.from_template("""
你是一个知识助手。请根据以下参考资料回答用户的问题。

参考资料：
{context}

用户问题：{question}

请基于以上参考资料回答，如果参考资料中没有相关信息，请如实告知。
""")

# 6. 初始化大模型
llm = ChatOpenAI(model="deepseek-v3", temperature=0)

# 7. 定义问答函数
def ask_question(question: str) -> str:
    # 检索相关文档
    docs = retriever.invoke(question)
    # 提取文档内容
    context = "\n\n".join([doc.page_content for doc in docs])
    # 构建提示词
    messages = prompt.format_messages(context=context, question=question)
    # 调用大模型
    response = llm.invoke(messages)
    return response.content

# 8. 测试提问
print("\n" + "=" * 50)
q1 = "什么是过拟合？"
print(f"📝 问题: {q1}")
a1 = ask_question(q1)
print(f"💡 答案: {a1}")

print("\n" + "=" * 50)
q2 = "机器学习有哪些类型？"
print(f"📝 问题: {q2}")
a2 = ask_question(q2)
print(f"💡 答案: {a2}")