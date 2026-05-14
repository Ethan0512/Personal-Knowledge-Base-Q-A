"""
RAG系统 Web界面 - 使用Streamlit构建
功能：
- 交互式聊天界面
- 文档上传和管理
- 实时问答
- 对话历史显示
"""
import streamlit as st
import os
from rag_optimized import OptimizedRAG


# 页面配置
st.set_page_config(
    page_title="智能RAG问答系统",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def load_rag_system():
    """加载RAG系统（缓存）"""
    return OptimizedRAG()


def init_session_state():
    """初始化会话状态"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False
    if 'use_rag' not in st.session_state:
        st.session_state.use_rag = True


def main():
    """主函数"""
    init_session_state()
    
    # 标题
    st.title("🤖 智能RAG问答系统")
    st.markdown("---")
    
    # 侧边栏 - 配置
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # API配置
        api_key = st.text_input(
            "API Key",
            value="sk-903c5d9bf6364d8bad6ebb5b08b439ef",
            type="password"
        )
        
        # 模型选择
        llm_model = st.selectbox(
            "大模型",
            ["deepseek-v3", "qwen-turbo", "qwen-plus"],
            index=0
        )
        
        embedding_model = st.selectbox(
            "Embedding模型",
            ["text-embedding-v4", "text-embedding-v3"],
            index=0
        )
        
        # RAG参数
        top_k = st.slider("检索文档数量 (Top-K)", 1, 10, 2)
        chunk_size = st.slider("文本块大小", 100, 1000, 200)
        temperature = st.slider("温度参数", 0.0, 1.0, 0.0)
        
        # 是否使用RAG
        use_rag = st.checkbox("启用RAG检索", value=True)
        st.session_state.use_rag = use_rag
        
        st.markdown("---")
        
        # 文档管理
        st.header("📄 文档管理")
        uploaded_file = st.file_uploader(
            "上传文本文件",
            type=['txt']
        )
        
        if uploaded_file is not None:
            # 保存上传的文件
            with open("uploaded_doc.txt", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ 文件已上传: {uploaded_file.name}")
            
            if st.button("🔄 重新加载文档"):
                with st.spinner('正在处理文档...'):
                    rag = load_rag_system()
                    rag.config["api_key"] = api_key
                    rag.config["llm_model"] = llm_model
                    rag.config["embedding_model"] = embedding_model
                    rag.config["top_k"] = top_k
                    rag.config["chunk_size"] = chunk_size
                    rag.config["temperature"] = temperature
                    
                    if rag.load_and_split("uploaded_doc.txt"):
                        if rag.generate_embeddings_batch():
                            st.session_state.rag_initialized = True
                            st.success("✅ 文档处理完成！")
                        else:
                            st.error("❌ 向量生成失败")
                    else:
                        st.error("❌ 文档加载失败")
        
        # 清空对话
        if st.button("🗑️ 清空对话历史"):
            st.session_state.messages = []
            st.rerun()
    
    # 主区域 - 聊天界面
    st.header("💬 智能对话")
    
    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 生成回答
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner('思考中...'):
                try:
                    rag = load_rag_system()
                    rag.config["api_key"] = api_key
                    rag.config["llm_model"] = llm_model
                    rag.config["embedding_model"] = embedding_model
                    rag.config["top_k"] = top_k
                    rag.config["temperature"] = temperature
                    
                    # 检查是否需要初始化
                    if not st.session_state.rag_initialized:
                        if os.path.exists("test.txt"):
                            if rag.load_and_split("test.txt"):
                                rag.generate_embeddings_batch()
                                st.session_state.rag_initialized = True
                    
                    # 生成回答
                    answer = rag.ask_question(prompt, use_rag=use_rag)
                    message_placeholder.markdown(answer)
                    
                    # 添加助手消息
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                    
                except Exception as e:
                    error_msg = f"❌ 错误: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # 底部信息
    st.markdown("---")
    st.caption(
        "💡 提示: 在侧边栏上传文档或配置参数 | "
        "支持Txt格式文档 | "
        "基于阿里云DashScope API"
    )


if __name__ == "__main__":
    main()
