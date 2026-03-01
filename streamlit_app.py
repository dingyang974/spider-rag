import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle
import faiss
from typing import List, Dict, Optional
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from loguru import logger

st.set_page_config(
    page_title="生育议题舆情智能分析与决策助手",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_vector_store():
    index_path = "./vector_store/index.faiss"
    docs_path = "./vector_store/documents.pkl"
    vectorizer_path = "./vector_store/vectorizer.pkl"
    
    if os.path.exists(index_path) and os.path.exists(docs_path):
        index = faiss.read_index(index_path)
        with open(docs_path, 'rb') as f:
            documents = pickle.load(f)
        vectorizer = None
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
        return index, documents, vectorizer
    return None, None, None


@st.cache_resource
def load_processed_data():
    data_path = "./vector_store/processed_data.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    base_url = st.secrets.get("OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    model = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "deepseek-chat"))
    
    if not api_key:
        return None, None, None
    
    return OpenAI(api_key=api_key, base_url=base_url), base_url, model


def tokenize_chinese(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    words = jieba.cut(text)
    return " ".join(words)


def search_similar_docs(query: str, index, documents: List[Dict], vectorizer, top_k: int = 10):
    if index is None or not documents or vectorizer is None:
        return []
    
    tokenized_query = tokenize_chinese(query)
    query_vec = vectorizer.transform([tokenized_query])
    query_embedding = query_vec.toarray().astype(np.float32)
    
    norm = np.linalg.norm(query_embedding)
    if norm > 0:
        query_embedding = query_embedding / norm
    
    scores, indices = index.search(query_embedding.reshape(1, -1), min(top_k * 2, len(documents)))
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(documents):
            doc = documents[idx].copy()
            doc['similarity_score'] = float(score)
            results.append(doc)
    
    return results[:top_k]


def generate_response(client, model, query: str, retrieved_docs: List[Dict]) -> str:
    if not client:
        return "请配置 API Key 后使用智能问答功能。"
    
    system_prompt = """你是一个专业的舆情分析助手，专注于生育议题的舆情分析。你的任务是基于提供的评论数据，为用户提供专业、客观、有洞察力的分析和建议。

你的回答应该遵循以下结构：
1. 主要讨论主题：总结用户问题的核心议题
2. 情绪分布特征：分析相关评论的情绪倾向
3. 风险识别：识别潜在的舆论风险点
4. 舆论趋势判断：分析舆论的发展趋势
5. 策略建议：提供可行的应对建议

请确保：
- 回答客观中立，基于事实
- 引用具体的评论内容作为依据
- 提供可操作的建议
- 语言简洁专业"""
    
    context_parts = ["以下是相关的评论数据：\n"]
    for i, doc in enumerate(retrieved_docs[:10], 1):
        context_parts.append(
            f"【评论{i}】\n"
            f"内容：{doc.get('content', '')}\n"
            f"点赞数：{doc.get('like_count', 0)}\n"
            f"相关度：{doc.get('similarity_score', 0):.3f}\n"
        )
    
    context = '\n'.join(context_parts)
    
    user_message = f"""用户问题：{query}

{context}

请基于以上评论数据，回答用户的问题。"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"生成回答时出错：{str(e)}"


def render_overview_page(df):
    st.markdown('<h1 class="main-header">📊 舆情总览</h1>', unsafe_allow_html=True)
    
    if df is None:
        st.warning("暂无数据，请先构建知识库")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总评论数", len(df))
    
    with col2:
        if 'sentiment' in df.columns:
            positive = len(df[df['sentiment'] == 'positive'])
            st.metric("正面评论", f"{positive} ({positive/len(df):.1%})")
        else:
            st.metric("正面评论", "N/A")
    
    with col3:
        if 'sentiment' in df.columns:
            negative = len(df[df['sentiment'] == 'negative'])
            st.metric("负面评论", f"{negative} ({negative/len(df):.1%})")
        else:
            st.metric("负面评论", "N/A")
    
    with col4:
        if 'sentiment' in df.columns:
            neutral = len(df[df['sentiment'] == 'neutral'])
            st.metric("中性评论", f"{neutral} ({neutral/len(df):.1%})")
        else:
            st.metric("中性评论", "N/A")
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🎭 情感分布")
        
        if 'sentiment' in df.columns:
            labels = ['正面', '负面', '中性']
            values = [
                len(df[df['sentiment'] == 'positive']),
                len(df[df['sentiment'] == 'negative']),
                len(df[df['sentiment'] == 'neutral'])
            ]
            colors = ['#2ecc71', '#e74c3c', '#95a5a6']
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                hole=0.4
            )])
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("暂无情感分析数据")
    
    with col_right:
        st.subheader("🔤 热门关键词")
        
        all_text = ' '.join([str(t) for t in df['content'].tolist() if pd.notna(t)])
        keywords = jieba.analyse.extract_tags(all_text, topK=15, withWeight=True)
        
        if keywords:
            kw_df = pd.DataFrame(keywords, columns=['word', 'weight'])
            fig_kw = px.bar(
                kw_df,
                x='weight',
                y='word',
                orientation='h',
                color='weight',
                color_continuous_scale='Blues'
            )
            fig_kw.update_layout(
                height=400,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("暂无关键词数据")


def render_qa_page(index, documents, vectorizer):
    st.markdown('<h1 class="main-header">💬 智能问答</h1>', unsafe_allow_html=True)
    
    client, base_url, llm_model = get_openai_client()
    
    if not client:
        st.warning("""
        ⚠️ 请先配置 API Key
        
        在 Streamlit Cloud 部署时，在 Settings → Secrets 中添加：
        ```
        OPENAI_API_KEY = "your-api-key"
        OPENAI_BASE_URL = "https://api.deepseek.com"
        OPENAI_MODEL = "deepseek-chat"
        ```
        """)
        return
    
    st.markdown("""
    ### 使用指南
    
    您可以向系统提问以下类型的问题：
    - **总结类**: 最近讨论焦点是什么？
    - **风险识别类**: 当前负面情绪主要集中在哪些方面？
    - **立场对比类**: 支持与反对的核心观点分别是什么？
    - **策略建议类**: 如果我是政策制定者，应如何回应？
    """)
    
    st.markdown("---")
    
    top_k = st.slider("检索数量", min_value=5, max_value=20, value=10)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 您:</strong> {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 助手:</strong><br>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    question = st.text_area("输入您的问题:", height=100)
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("发送问题", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("正在分析..."):
                    retrieved_docs = search_similar_docs(question, index, documents, vectorizer, top_k)
                    answer = generate_response(client, llm_model, question, retrieved_docs)
                    
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })
                    st.rerun()
            else:
                st.warning("请输入问题")
    
    with col_btn2:
        if st.button("清空对话", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    st.markdown("---")
    st.subheader("🎯 快捷问题")
    
    quick_questions = [
        "最近讨论焦点是什么？",
        "当前负面情绪主要集中在哪些方面？",
        "支持与反对的核心观点分别是什么？",
        "如果我是政策制定者，应如何回应？"
    ]
    
    cols = st.columns(2)
    for i, q in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                with st.spinner("正在分析..."):
                    retrieved_docs = search_similar_docs(q, index, documents, vectorizer, top_k)
                    answer = generate_response(client, llm_model, q, retrieved_docs)
                    
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": q
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })
                    st.rerun()


def render_comments_page(df):
    st.markdown('<h1 class="main-header">📝 评论浏览</h1>', unsafe_allow_html=True)
    
    if df is None:
        st.warning("暂无数据")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        sentiment_filter = st.selectbox(
            "情感筛选",
            options=["全部", "正面", "负面", "中性"],
            index=0
        )
    
    with col2:
        page_size = st.selectbox("每页显示", options=[10, 20, 50], index=1)
    
    filtered_df = df.copy()
    
    if sentiment_filter != "全部" and 'sentiment' in filtered_df.columns:
        sentiment_map = {"正面": "positive", "负面": "negative", "中性": "neutral"}
        filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_map[sentiment_filter]]
    
    if 'like_count' in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by='like_count', ascending=False)
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    
    total = len(filtered_df)
    start_idx = (st.session_state.current_page - 1) * page_size
    end_idx = start_idx + page_size
    
    st.markdown(f"**共 {total} 条评论**")
    
    page_df = filtered_df.iloc[start_idx:end_idx]
    
    for idx, row in page_df.iterrows():
        sentiment = row.get('sentiment', 'neutral')
        sentiment_emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}
        sentiment_color = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
        
        emoji = sentiment_emoji.get(sentiment, "😐")
        color = sentiment_color.get(sentiment, "#95a5a6")
        
        st.markdown(f"""
        <div style="border-left: 4px solid {color}; padding-left: 1rem; margin: 1rem 0;">
            <p>{row.get('content', '')}</p>
            <small>
                {emoji} {sentiment} | 
                👍 {row.get('like_count', 0)} | 
                📅 {row.get('publish_time', '')}
            </small>
        </div>
        """, unsafe_allow_html=True)
    
    total_pages = (total + page_size - 1) // page_size
    
    col_prev, col_page, col_next = st.columns([1, 2, 1])
    
    with col_prev:
        if st.button("上一页", disabled=st.session_state.current_page <= 1):
            st.session_state.current_page -= 1
            st.rerun()
    
    with col_page:
        st.markdown(f"<p style='text-align: center;'>第 {st.session_state.current_page} / {total_pages} 页</p>", unsafe_allow_html=True)
    
    with col_next:
        if st.button("下一页", disabled=st.session_state.current_page >= total_pages):
            st.session_state.current_page += 1
            st.rerun()


def main():
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2>📊 舆情助手</h2>
        <p>生育议题智能分析</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "导航",
        options=["📈 舆情总览", "💬 智能问答", "📝 评论浏览"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    with st.spinner("正在加载数据..."):
        index, documents, vectorizer = load_vector_store()
        df = load_processed_data()
    
    if index is not None:
        st.sidebar.success(f"✅ 已加载 {len(documents)} 条数据")
    else:
        st.sidebar.warning("⚠️ 知识库未加载")
    
    if "📈 舆情总览" in page:
        render_overview_page(df)
    elif "💬 智能问答" in page:
        render_qa_page(index, documents, vectorizer)
    elif "📝 评论浏览" in page:
        render_comments_page(df)


if __name__ == "__main__":
    main()
