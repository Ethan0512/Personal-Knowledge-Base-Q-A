"""
RAG系统 - 高级功能版
功能：
- 支持多种文档格式（PDF、Word、TXT）
- 对话历史记忆
- 流式输出
- 更智能的检索
"""
import dashscope
from dashscope import TextEmbedding
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import os
import numpy as np
from typing import List, Dict, Optional
import time


class AdvancedRAG:
    """高级RAG系统"""
    
    def __init__(self, config: Dict = None):
        """初始化高级RAG系统"""
        default_config = {
            "api_key": "sk-903c5d9bf6364d8bad6ebb5b08b439ef",
            "embedding_model": "text-embedding-v4",
            "llm_model": "deepseek-v3",
            "chunk_size": 200,
            "chunk_overlap": 50,
            "top_k": 2,
            "temperature": 0,
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "max_history": 10  # 最大对话历史数
        }
        
        self.config = {**default_config, **(config or {})}
        
        # 设置API
        dashscope.api_key = self.config["api_key"]
        os.environ["OPENAI_API_KEY"] = self.config["api_key"]
        os.environ["OPENAI_API_BASE"] = self.config["api_base"]
        
        # 初始化存储
        self.embeddings_list = []
        self.texts = []
        self.metadata = []  # 存储文档元数据
        self.is_initialized = False
        
        # 对话历史
        self.conversation_history = []
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=self.config["llm_model"],
            temperature=self.config["temperature"],
            streaming=True  # 启用流式输出
        )
        
        print("✅ 高级RAG系统初始化完成")
    
    def load_document(self, file_path: str) -> bool:
        """
        加载文档（自动检测格式）
        
        Args:
            file_path: 文档路径
            
        Returns:
            是否成功
        """
        try:
            ext = os.path.splitext(file_path)[1].lower()
            print(f"📄 正在加载文档: {file_path}")
            
            # 根据文件扩展名选择加载器
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.docx':
                loader = Docx2txtLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                print(f"❌ 不支持的文件格式: {ext}")
                return False
            
            documents = loader.load()
            print(f"✅ 加载了 {len(documents)} 个文档页面")
            
            # 切分文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config["chunk_size"],
                chunk_overlap=self.config["chunk_overlap"]
            )
            docs = text_splitter.split_documents(documents)
            
            self.texts = [doc.page_content for doc in docs]
            self.metadata = [doc.metadata for doc in docs]
            
            print(f"✅ 文档已切分为 {len(self.texts)} 个片段")
            return True
            
        except Exception as e:
            print(f"❌ 文档加载失败: {str(e)}")
            return False
    
    def generate_embeddings_batch(self, batch_size: int = 10) -> bool:
        """批量生成向量嵌入"""
        if not self.texts:
            print("❌ 没有文本可处理")
            return False
        
        print(f"⏳ 正在批量生成向量...")
        self.embeddings_list = []
        total = len(self.texts)
        
        for i in range(0, total, batch_size):
            batch = self.texts[i:i + batch_size]
            try:
                resp = TextEmbedding.call(
                    model=self.config["embedding_model"],
                    input=batch
                )
                
                if resp.status_code == 200:
                    embeddings = [item["embedding"] for item in resp.output["embeddings"]]
                    self.embeddings_list.extend(embeddings)
                    print(f"   进度: {min(i + batch_size, total)}/{total}")
                else:
                    print(f"❌ 批次失败: {resp.message}")
                    return False
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ 生成向量时出错: {str(e)}")
                return False
        
        self.is_initialized = True
        print(f"✅ 成功生成 {len(self.embeddings_list)} 个向量")
        return True
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        检索相关文档（返回带元数据的结果）
        
        Returns:
            包含文本和元数据的字典列表
        """
        if not self.is_initialized:
            return []
        
        top_k = top_k or self.config["top_k"]
        
        try:
            # 获取查询向量
            resp = TextEmbedding.call(
                model=self.config["embedding_model"],
                input=query
            )
            
            if resp.status_code != 200:
                return []
            
            query_vec = resp.output["embeddings"][0]["embedding"]
            
            # 计算相似度
            similarities = []
            for doc_vec in self.embeddings_list:
                dot = np.dot(query_vec, doc_vec)
                norm_q = np.linalg.norm(query_vec)
                norm_d = np.linalg.norm(doc_vec)
                
                sim = dot / (norm_q * norm_d) if norm_q and norm_d else 0
                similarities.append(sim)
            
            # 取top_k
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    "content": self.texts[idx],
                    "metadata": self.metadata[idx] if idx < len(self.metadata) else {},
                    "similarity": similarities[idx]
                })
            
            return results
            
        except Exception as e:
            print(f"❌ 检索失败: {str(e)}")
            return []
    
    def ask_with_history(self, question: str, use_rag: bool = True) -> str:
        """
        带对话历史的问答
        
        Args:
            question: 用户问题
            use_rag: 是否使用RAG
            
        Returns:
            模型回答
        """
        try:
            # 检索相关文档
            context = ""
            if use_rag and self.is_initialized:
                docs = self.retrieve(question)
                context = "\n\n".join([doc["content"] for doc in docs])
            
            # 构建提示词（包含对话历史）
            history_text = ""
            if self.conversation_history:
                history_text = "\n\n对话历史:\n"
                for msg in self.conversation_history[-self.config["max_history"]:]:
                    role = "用户" if isinstance(msg, HumanMessage) else "助手"
                    history_text += f"{role}: {msg.content}\n"
            
            prompt = ChatPromptTemplate.from_template("""
你是一个知识助手。请根据以下参考资料和对话历史回答用户的问题。

参考资料：
{context}

{history}

当前问题：{question}

请基于参考资料和对话历史回答，如果没有相关信息，请如实告知。
""")
            
            # 生成回答
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "context": context,
                "history": history_text,
                "question": question
            })
            
            # 更新对话历史
            self.conversation_history.append(HumanMessage(content=question))
            self.conversation_history.append(AIMessage(content=response))
            
            return response
            
        except Exception as e:
            print(f"❌ 生成回答失败: {str(e)}")
            return f"抱歉，出现错误: {str(e)}"
    
    def stream_response(self, question: str, use_rag: bool = True):
        """
        流式输出回答（实时显示）
        
        Yields:
            文本片段
        """
        try:
            # 检索
            context = ""
            if use_rag and self.is_initialized:
                docs = self.retrieve(question)
                context = "\n\n".join([doc["content"] for doc in docs])
            
            # 构建提示词
            prompt = ChatPromptTemplate.from_template("""
你是一个知识助手。请根据以下参考资料回答用户的问题。

参考资料：
{context}

用户问题：{question}

请基于参考资料回答。
""")
            
            # 流式调用
            chain = prompt | self.llm | StrOutputParser()
            
            for chunk in chain.stream({"context": context, "question": question}):
                yield chunk
                
        except Exception as e:
            yield f"\n❌ 错误: {str(e)}"
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        print("✅ 对话历史已清空")
    
    def get_history_summary(self) -> str:
        """获取对话历史摘要"""
        if not self.conversation_history:
            return "暂无对话历史"
        
        summary = f"对话轮数: {len(self.conversation_history) // 2}\n"
        summary += "最近对话:\n"
        
        for msg in self.conversation_history[-4:]:
            role = "👤 用户" if isinstance(msg, HumanMessage) else "🤖 助手"
            preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            summary += f"{role}: {preview}\n"
        
        return summary


def main():
    """主函数 - 演示高级功能"""
    print("=" * 60)
    print("🚀 高级RAG系统演示")
    print("=" * 60)
    
    # 创建系统
    rag = AdvancedRAG()
    
    # 测试不同格式的文档
    test_files = ["test.txt", "test.pdf", "test.docx"]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n{'='*60}")
            print(f"测试文件: {file_path}")
            
            if rag.load_document(file_path):
                if rag.generate_embeddings_batch():
                    # 测试带历史的对话
                    print("\n--- 第一轮对话 ---")
                    q1 = "什么是过拟合？"
                    print(f"📝 问题: {q1}")
                    a1 = rag.ask_with_history(q1)
                    print(f"💡 答案: {a1[:200]}...")
                    
                    print("\n--- 第二轮对话（带历史）---")
                    q2 = "如何防止它？"
                    print(f"📝 问题: {q2}")
                    a2 = rag.ask_with_history(q2)
                    print(f"💡 答案: {a2[:200]}...")
                    
                    # 显示历史摘要
                    print("\n" + rag.get_history_summary())
                    
                    # 测试流式输出
                    print("\n--- 流式输出测试 ---")
                    print("📝 问题: 总结一下")
                    print("💡 答案: ", end="", flush=True)
                    
                    for chunk in rag.stream_response("总结一下主要内容"):
                        print(chunk, end="", flush=True)
                    
                    print("\n")
                    break
    
    print("\n✅ 高级功能演示完成！")


if __name__ == "__main__":
    main()
