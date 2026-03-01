import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE_URL = "http://localhost:8000"

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
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
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


def check_api_status() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_overview() -> Optional[Dict]:
    try:
        response = requests.get(f"{API_BASE_URL}/api/overview", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"获取概览数据失败: {e}")
    return None


def build_knowledge_base(data_path: str) -> Dict:
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/build-knowledge-base",
            json={"data_path": data_path},
            timeout=300
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


def query_rag(question: str, sentiment_filter: str = None, top_k: int = 10) -> Dict:
    try:
        payload = {
            "question": question,
            "top_k": top_k
        }
        if sentiment_filter:
            payload["sentiment_filter"] = sentiment_filter
        
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json=payload,
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"answer": f"查询失败: {e}", "sources": []}


def get_sentiment_trend(freq: str = "D") -> Optional[Dict]:
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/sentiment-trend",
            params={"freq": freq},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"获取情感趋势失败: {e}")
    return None


def get_comments(page: int = 1, page_size: int = 20, sentiment: str = None) -> Optional[Dict]:
    try:
        params = {"page": page, "page_size": page_size}
        if sentiment:
            params["sentiment"] = sentiment
        
        response = requests.get(
            f"{API_BASE_URL}/api/comments",
            params=params,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"获取评论失败: {e}")
    return None


def identify_risks(topic: str = None) -> Optional[Dict]:
    try:
        params = {}
        if topic:
            params["topic"] = topic
        
        response = requests.get(
            f"{API_BASE_URL}/api/risks",
            params=params,
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"风险识别失败: {e}")
    return None


def compare_viewpoints(topic: str) -> Optional[Dict]:
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/viewpoints/{topic}",
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"观点对比失败: {e}")
    return None


def generate_strategy(context: str, role: str = "政策制定者") -> Optional[Dict]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/strategy",
            json={"context": context, "role": role},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"策略生成失败: {e}")
    return None


def render_overview_page():
    st.markdown('<h1 class="main-header">📊 舆情总览</h1>', unsafe_allow_html=True)
    
    overview = get_overview()
    
    if not overview:
        st.warning("请先构建知识库")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总评论数", overview["total_comments"])
    
    with col2:
        sentiment_dist = overview["sentiment_distribution"]
        st.metric("正面评论", f"{sentiment_dist['positive']} ({sentiment_dist['positive_ratio']:.1%})")
    
    with col3:
        st.metric("负面评论", f"{sentiment_dist['negative']} ({sentiment_dist['negative_ratio']:.1%})")
    
    with col4:
        st.metric("中性评论", f"{sentiment_dist['neutral']} ({sentiment_dist['neutral_ratio']:.1%})")
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🎭 情感分布")
        
        labels = ['正面', '负面', '中性']
        values = [
            sentiment_dist['positive'],
            sentiment_dist['negative'],
            sentiment_dist['neutral']
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
    
    with col_right:
        st.subheader("📈 情感趋势")
        
        trend_data = get_sentiment_trend()
        if trend_data and trend_data.get("trend"):
            trend_df = pd.DataFrame(trend_data["trend"])
            trend_df['date'] = pd.to_datetime(trend_df['date'])
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=trend_df['date'],
                y=trend_df['sentiment_balance'],
                mode='lines+markers',
                name='情感平衡值',
                line=dict(color='#3498db', width=2)
            ))
            fig_trend.update_layout(
                xaxis_title='日期',
                yaxis_title='情感平衡值',
                height=400
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("暂无情感趋势数据")
    
    st.markdown("---")
    
    col_topic, col_keyword = st.columns(2)
    
    with col_topic:
        st.subheader("📌 主要讨论主题")
        
        topics = overview.get("topics", [])
        if topics:
            for topic in topics:
                with st.expander(topic["label"], expanded=False):
                    st.write("**关键词:**", ", ".join(topic["keywords"]))
        else:
            st.info("暂无主题数据")
    
    with col_keyword:
        st.subheader("🔤 热门关键词")
        
        keywords = overview.get("top_keywords", [])
        if keywords:
            kw_df = pd.DataFrame(keywords[:15])
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


def render_qa_page():
    st.markdown('<h1 class="main-header">💬 智能问答</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 使用指南
    
    您可以向系统提问以下类型的问题：
    - **总结类**: 最近讨论焦点是什么？
    - **风险识别类**: 当前负面情绪主要集中在哪些方面？
    - **立场对比类**: 支持与反对的核心观点分别是什么？
    - **策略建议类**: 如果我是政策制定者，应如何回应？
    """)
    
    st.markdown("---")
    
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        sentiment_filter = st.selectbox(
            "情感筛选",
            options=["全部", "正面", "负面", "中性"],
            index=0
        )
        sentiment_map = {"全部": None, "正面": "positive", "负面": "negative", "中性": "neutral"}
    
    with col_filter2:
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
                    result = query_rag(
                        question,
                        sentiment_filter=sentiment_map.get(sentiment_filter),
                        top_k=top_k
                    )
                    
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question
                    })
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result.get("answer", "抱歉，无法生成回答")
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
                    result = query_rag(q, top_k=top_k)
                    
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": q
                    })
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result.get("answer", "抱歉，无法生成回答")
                    })
                    
                    st.rerun()


def render_comments_page():
    st.markdown('<h1 class="main-header">📝 评论浏览</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        sentiment_filter = st.selectbox(
            "情感筛选",
            options=["全部", "正面", "负面", "中性"],
            index=0
        )
        sentiment_map = {"全部": None, "正面": "positive", "负面": "negative", "中性": "neutral"}
    
    with col2:
        page_size = st.selectbox("每页显示", options=[10, 20, 50], index=1)
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    
    comments_data = get_comments(
        page=st.session_state.current_page,
        page_size=page_size,
        sentiment=sentiment_map.get(sentiment_filter)
    )
    
    if comments_data:
        st.markdown(f"**共 {comments_data['total']} 条评论**")
        
        for comment in comments_data["comments"]:
            sentiment_emoji = {
                "positive": "😊",
                "negative": "😠",
                "neutral": "😐"
            }
            sentiment_color = {
                "positive": "#2ecc71",
                "negative": "#e74c3c",
                "neutral": "#95a5a6"
            }
            
            emoji = sentiment_emoji.get(comment["sentiment"], "😐")
            color = sentiment_color.get(comment["sentiment"], "#95a5a6")
            
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding-left: 1rem; margin: 1rem 0;">
                <p>{comment["content"]}</p>
                <small>
                    {emoji} {comment["sentiment"]} | 
                    👍 {comment["like_count"]} | 
                    📅 {comment["publish_time"]}
                </small>
            </div>
            """, unsafe_allow_html=True)
        
        total_pages = (comments_data["total"] + page_size - 1) // page_size
        
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
    else:
        st.info("暂无评论数据")


def render_analysis_page():
    st.markdown('<h1 class="main-header">🔍 深度分析</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["⚠️ 风险识别", "⚖️ 观点对比", "📋 策略建议"])
    
    with tab1:
        st.subheader("风险识别")
        
        topic_input = st.text_input("指定主题（可选）:", key="risk_topic")
        
        if st.button("识别风险", type="primary"):
            with st.spinner("正在分析风险..."):
                result = identify_risks(topic_input if topic_input else None)
                
                if result:
                    st.markdown("### 风险分析结果")
                    st.markdown(result.get("risks", ""))
                    
                    if result.get("negative_sources"):
                        st.markdown("#### 相关负面评论")
                        for i, source in enumerate(result["negative_sources"][:5], 1):
                            st.markdown(f"""
                            <div style="background-color: #fff5f5; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;">
                                <strong>{i}.</strong> {source.get("content", "")}
                                <br><small>相似度: {source.get("similarity_score", 0):.3f}</small>
                            </div>
                            """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("观点对比")
        
        compare_topic = st.text_input("输入对比主题:", key="compare_topic")
        
        if st.button("对比观点", type="primary"):
            if compare_topic:
                with st.spinner("正在分析观点..."):
                    result = compare_viewpoints(compare_topic)
                    
                    if result:
                        col_support, col_oppose = st.columns(2)
                        
                        with col_support:
                            st.markdown("### ✅ 支持观点")
                            st.markdown(result.get("supporting_views", ""))
                        
                        with col_oppose:
                            st.markdown("### ❌ 反对观点")
                            st.markdown(result.get("opposing_views", ""))
            else:
                st.warning("请输入对比主题")
    
    with tab3:
        st.subheader("策略建议")
        
        strategy_context = st.text_area("输入背景情况:", height=100, key="strategy_context")
        strategy_role = st.selectbox("角色设定:", ["政策制定者", "公关人员", "研究人员"])
        
        if st.button("生成策略", type="primary"):
            if strategy_context:
                with st.spinner("正在生成策略..."):
                    result = generate_strategy(strategy_context, strategy_role)
                    
                    if result:
                        st.markdown("### 📋 策略建议")
                        st.markdown(f"**角色:** {result.get('role', '')}")
                        st.markdown(f"**背景:** {result.get('context', '')}")
                        st.markdown("---")
                        st.markdown(result.get("strategy", ""))
            else:
                st.warning("请输入背景情况")


def render_settings_page():
    st.markdown('<h1 class="main-header">⚙️ 系统设置</h1>', unsafe_allow_html=True)
    
    st.subheader("API 状态")
    
    if check_api_status():
        st.success("✅ API 服务正常运行")
    else:
        st.error("❌ API 服务未运行，请先启动后端服务")
    
    st.markdown("---")
    
    st.subheader("知识库管理")
    
    data_path = st.text_input("数据文件路径:", value="./data/comments.csv")
    
    if st.button("构建知识库", type="primary"):
        with st.spinner("正在构建知识库，这可能需要几分钟..."):
            result = build_knowledge_base(data_path)
            
            if result.get("success"):
                st.success(f"✅ {result.get('message')}，共处理 {result.get('documents_count', 0)} 条文档")
            else:
                st.error(f"❌ 构建失败: {result.get('message', '未知错误')}")
    
    st.markdown("---")
    
    st.subheader("系统信息")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/statistics", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**数据统计**")
                st.json({
                    "总评论数": stats.get("total_comments", 0),
                    "情感分布": stats.get("sentiment_counts", {}),
                    "点赞统计": stats.get("like_statistics", {})
                })
            
            with col2:
                st.markdown("**向量库信息**")
                st.json(stats.get("vector_store", {}))
    except:
        st.info("无法获取系统统计信息")


def main():
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2>📊 舆情助手</h2>
        <p>生育议题智能分析</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "导航",
        options=["📈 舆情总览", "💬 智能问答", "📝 评论浏览", "🔍 深度分析", "⚙️ 系统设置"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    if check_api_status():
        st.sidebar.success("✅ 服务正常")
    else:
        st.sidebar.error("❌ 服务离线")
    
    st.sidebar.markdown("""
    ---
    **使用说明**
    
    1. 首先在"系统设置"中构建知识库
    2. 然后可以查看舆情总览
    3. 使用智能问答进行交互
    """)
    
    if "📈 舆情总览" in page:
        render_overview_page()
    elif "💬 智能问答" in page:
        render_qa_page()
    elif "📝 评论浏览" in page:
        render_comments_page()
    elif "🔍 深度分析" in page:
        render_analysis_page()
    elif "⚙️ 系统设置" in page:
        render_settings_page()


if __name__ == "__main__":
    main()
