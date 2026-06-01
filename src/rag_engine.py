from typing import List, Dict, Optional
from loguru import logger
from openai import OpenAI
from config import settings
from .vector_store import VectorStore


class RAGEngine:
    
    SYSTEM_PROMPT = """你是 InsightOps 企业市场情报 Agent，服务对象是新消费品牌的品牌运营、公关风控、营销增长和竞品策略团队。你的任务是基于提供的评论证据，为企业内部运营人员提供专业、客观、有洞察力且可执行的市场情报分析。

你的回答应该遵循以下结构：
1. 关键结论：先给出可直接用于业务判断的结论
2. 证据摘要：引用评论证据，说明用户主要反馈
3. 风险/机会判断：识别潜在公关风险、营销优化点或竞品机会
4. 建议动作：给出可分派给具体角色的下一步行动
5. 可交付物：按需输出回应口径、日报摘要、活动复盘、竞品话术或证据清单

请确保：
- 回答基于证据，不编造不存在的数据
- 引用具体评论内容作为依据
- 明确区分事实、推断和建议
- 语言简洁专业，适合放入企业内部报告或 Copilot 输出"""

    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store
        self.client = None
        if settings.LLM_API_KEY:
            self.client = OpenAI(
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL
            )
        self.model = settings.LLM_MODEL
    
    def build_context(self, retrieved_docs: List[Dict]) -> str:
        if not retrieved_docs:
            return "暂无相关评论数据。"
        
        context_parts = ["以下是相关的评论数据：\n"]
        
        for i, doc in enumerate(retrieved_docs, 1):
            sentiment_map = {
                "positive": "正面",
                "negative": "负面",
                "neutral": "中性"
            }
            sentiment_cn = sentiment_map.get(doc.get('sentiment', 'neutral'), '中性')
            
            context_parts.append(
                f"【评论{i}】\n"
                f"内容：{doc.get('content', '')}\n"
                f"情感：{sentiment_cn}（置信度：{doc.get('sentiment_score', 0):.2f}）\n"
                f"点赞数：{doc.get('like_count', 0)}\n"
                f"发布时间：{doc.get('publish_time', '未知')}\n"
                f"相关度：{doc.get('similarity_score', 0):.3f}\n"
            )
        
        return '\n'.join(context_parts)
    
    def generate_response(self, 
                         query: str,
                         retrieved_docs: List[Dict],
                         conversation_history: List[Dict] = None) -> str:
        
        context = self.build_context(retrieved_docs)
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]
        
        if conversation_history:
            messages.extend(conversation_history[-6:])
        
        user_message = f"""用户问题：{query}

{context}

请基于以上评论数据，回答用户的问题。"""
        
        messages.append({"role": "user", "content": user_message})

        if self.client is None:
            return "DeepSeek API Key 未配置。请设置 DEEPSEEK_API_KEY 后重启后端服务。"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=settings.MAX_TOKENS,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"
    
    def query(self,
              question: str,
              top_k: int = None,
              sentiment_filter: Optional[str] = None,
              min_likes: int = 0,
              conversation_history: List[Dict] = None) -> Dict:
        
        if self.vector_store is None:
            return {
                "answer": "向量库未初始化，请先构建知识库。",
                "sources": [],
                "error": "Vector store not initialized"
            }
        
        retrieved_docs = self.vector_store.search(
            query=question,
            top_k=top_k or settings.TOP_K_RETRIEVAL,
            sentiment_filter=sentiment_filter,
            min_likes=min_likes
        )
        
        answer = self.generate_response(
            query=question,
            retrieved_docs=retrieved_docs,
            conversation_history=conversation_history
        )
        
        return {
            "answer": answer,
            "sources": retrieved_docs,
            "query": question,
            "retrieval_count": len(retrieved_docs)
        }
    
    def summarize_topic(self, 
                       topic: str,
                       time_range: Optional[tuple] = None) -> Dict:
        
        query = f"关于{topic}的讨论，请总结主要观点和情绪倾向"
        
        result = self.query(query, top_k=15)
        
        result["topic"] = topic
        result["time_range"] = time_range
        
        return result
    
    def identify_risks(self, 
                      topic: Optional[str] = None) -> Dict:
        
        query = "当前存在哪些负面情绪和潜在风险？"
        if topic:
            query = f"关于{topic}，存在哪些负面情绪和潜在风险？"
        
        result = self.query(
            query,
            top_k=15,
            sentiment_filter="negative"
        )
        
        return {
            "risks": result["answer"],
            "negative_sources": result["sources"],
            "query": query
        }
    
    def compare_viewpoints(self, topic: str) -> Dict:
        
        positive_query = f"关于{topic}，支持的观点有哪些？"
        negative_query = f"关于{topic}，反对的观点有哪些？"
        
        positive_result = self.query(
            positive_query,
            top_k=10,
            sentiment_filter="positive"
        )
        
        negative_result = self.query(
            negative_query,
            top_k=10,
            sentiment_filter="negative"
        )
        
        return {
            "topic": topic,
            "supporting_views": positive_result["answer"],
            "supporting_sources": positive_result["sources"],
            "opposing_views": negative_result["answer"],
            "opposing_sources": negative_result["sources"]
        }
    
    def generate_strategy(self, 
                         context: str,
                         role: str = "政策制定者") -> Dict:
        
        query = f"""如果我是{role}，针对以下情况，应该如何回应？

{context}

请提供具体的策略建议。"""
        
        result = self.query(query, top_k=10)
        
        return {
            "role": role,
            "context": context,
            "strategy": result["answer"],
            "sources": result["sources"]
        }
