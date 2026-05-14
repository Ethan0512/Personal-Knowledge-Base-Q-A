"""
本地大模型部署方案 - 使用Ollama
功能：
- 本地运行开源大模型（Llama3、Qwen等）
- 无需API密钥，完全离线
- 支持RAG集成
"""
import ollama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import numpy as np
from typing import List


class LocalRAG:
    """本地RAG系统 - 基于Ollama"""
    
    def __init__(self, model_name: str = "qwen2.5:7b"):
        """
        初始化本地RAG系统
        
        Args:
            model_name: Ollama模型名称，如 "llama3.2", "qwen2.5:7b"
        """
        self.model_name = model_name
        self.embeddings_list = []
        self.texts = []
        self.is_initialized = False
        
        # 初始化Embedding模型
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        
        # 初始化LLM
        self.llm = ChatOllama(
            model=model_name,
            temperature=0
        )
        
        print(f"✅ 本地RAG系统初始化完成 (模型: {model_name})")
    
    def check_ollama_running(self) -> bool:
        """检查Ollama是否正在运行"""
        try:
            ollama.list()
            return True
        except Exception as e:
            print(f"❌ Ollama未运行或无法连接: {str(e)}")
            print("\n💡 请先安装并启动Ollama:")
            print("   1. 访问 https://ollama.com 下载安装")
            print("   2. 运行命令: ollama serve")
            print(f"   3. 拉取模型: ollama pull {self.model_name}")
            return False
    
    def load_and_split(self, file_path: str, chunk_size: int = 200, 
                      chunk_overlap: int = 50) -> bool:
        """加载并切分文档"""
        try:
            print(f"📄 正在加载文档: {file_path}")
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            print(f"✅ 加载了 {len(documents)} 个文档")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            docs = text_splitter.split_documents(documents)
            self.texts = [doc.page_content for doc in docs]
            print(f"✅ 文档已切分为 {len(self.texts)} 个片段")
            
            return True
        except Exception as e:
            print(f"❌ 文档加载失败: {str(e)}")
            return False
    
    def generate_embeddings(self) -> bool:
        """生成本地向量嵌入"""
        if not self.texts:
            print("❌ 没有文本可处理")
            return False
        
        print(f"⏳ 正在生成本地向量... ({len(self.texts)} 个文本)")
        
        try:
            # 使用Ollama Embedding
            self.embeddings_list = self.embedding_model.embed_documents(self.texts)
            self.is_initialized = True
            
            print(f"✅ 成功生成 {len(self.embeddings_list)} 个向量，维度 {len(self.embeddings_list[0])}")
            return True
            
        except Exception as e:
            print(f"❌ 生成向量失败: {str(e)}")
            return False
    
    def create_vector_store(self) -> Chroma:
        """创建向量数据库"""
        if not self.is_initialized:
            print("❌ 请先生成向量")
            return None
        
        texts = self.texts
        vectorstore = Chroma.from_texts(texts, self.embedding_model)
        print("✅ 向量数据库创建完成")
        
        return vectorstore
    
    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        """检索相关文档"""
        if not self.is_initialized:
            print("❌ 系统未初始化")
            return []
        
        try:
            # 生成查询向量
            query_vec = self.embedding_model.embed_query(query)
            
            # 计算相似度
            similarities = []
            for doc_vec in self.embeddings_list:
                dot = np.dot(query_vec, doc_vec)
                norm_q = np.linalg.norm(query_vec)
                norm_d = np.linalg.norm(doc_vec)
                
                if norm_q == 0 or norm_d == 0:
                    sim = 0
                else:
                    sim = dot / (norm_q * norm_d)
                similarities.append(sim)
            
            # 取top_k
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = [self.texts[i] for i in top_indices]
            
            return results
            
        except Exception as e:
            print(f"❌ 检索失败: {str(e)}")
            return []
    
    def ask_question(self, question: str, use_rag: bool = True) -> str:
        """回答问题"""
        try:
            if use_rag and self.is_initialized:
                # 使用RAG
                docs = self.retrieve(question)
                context = "\n\n".join(docs)
                
                prompt = ChatPromptTemplate.from_template("""
你是一个知识助手。请根据以下参考资料回答用户的问题。

参考资料：
{context}

用户问题：{question}

请基于参考资料回答，如果没有相关信息，请如实告知。
""")
            else:
                # 直接问答
                context = "无参考资料"
                prompt = ChatPromptTemplate.from_template("""
你是一个知识助手。请回答用户的问题。

用户问题：{question}
""")
            
            # 构建链
            chain = prompt | self.llm | StrOutputParser()
            
            # 生成回答
            response = chain.invoke({"context": context, "question": question})
            return response
            
        except Exception as e:
            print(f"❌ 生成回答失败: {str(e)}")
            return f"抱歉，出现错误: {str(e)}"
    
    def chat_stream(self, question: str):
        """流式输出对话（实时显示）"""
        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': question}],
                stream=True
            )
            
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
                
        except Exception as e:
            print(f"\n❌ 流式对话失败: {str(e)}")


def setup_instructions():
    """打印设置说明"""
    print("=" * 60)
    print("📦 本地大模型部署指南")
    print("=" * 60)
    print()
    print("步骤1: 安装Ollama")
    print("  - Windows: 访问 https://ollama.com/download/windows")
    print("  - 下载并安装Ollama")
    print()
    print("步骤2: 启动Ollama服务")
    print("  - 打开命令行，运行: ollama serve")
    print("  - 或在后台运行Ollama应用")
    print()
    print("步骤3: 拉取模型（选择一个）")
    print("  - Qwen2.5 (推荐中文): ollama pull qwen2.5:7b")
    print("  - Llama3.2: ollama pull llama3.2")
    print("  - Phi3 (轻量级): ollama pull phi3")
    print()
    print("步骤4: 安装Python依赖")
    print("  pip install ollama langchain langchain-community chromadb")
    print()
    print("步骤5: 运行本脚本")
    print("  python rag_local.py")
    print()
    print("=" * 60)


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 本地RAG系统演示")
    print("=" * 60)
    
    # 检查Ollama
    rag = LocalRAG(model_name="qwen2.5:7b")
    
    if not rag.check_ollama_running():
        setup_instructions()
        return
    
    # 加载文档
    if os.path.exists("test.txt"):
        if rag.load_and_split("test.txt"):
            if rag.generate_embeddings():
                print("\n" + "=" * 60)
                
                # 测试问答
                questions = [
                    "什么是过拟合？",
                    "机器学习有哪些类型？"
                ]
                
                for question in questions:
                    print(f"\n📝 问题: {question}")
                    answer = rag.ask_question(question, use_rag=True)
                    print(f"💡 答案: {answer}")
                    print("-" * 60)
                
                print("\n✅ 本地RAG演示完成！")
    else:
        print("❌ 找不到 test.txt 文件")


if __name__ == "__main__":
    main()
