from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 10
    sentiment_filter: Optional[str] = None
    min_likes: Optional[int] = 0
    conversation_history: Optional[List[Dict[str, str]]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    retrieval_count: int


class SentimentDistribution(BaseModel):
    positive: int
    negative: int
    neutral: int
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float


class TopicInfo(BaseModel):
    topic_id: int
    keywords: List[str]
    label: str


class OverviewResponse(BaseModel):
    total_comments: int
    sentiment_distribution: SentimentDistribution
    topics: List[TopicInfo]
    time_range: Optional[Dict[str, str]]
    top_keywords: List[Dict[str, Any]]


class CommentItem(BaseModel):
    id: int
    content: str
    sentiment: str
    sentiment_score: float
    like_count: int
    publish_time: Optional[str]


class CommentsResponse(BaseModel):
    comments: List[CommentItem]
    total: int
    page: int
    page_size: int


class BuildKnowledgeBaseRequest(BaseModel):
    data_path: Optional[str] = None
    force_rebuild: Optional[bool] = False


class BuildKnowledgeBaseResponse(BaseModel):
    success: bool
    message: str
    documents_count: int


class RiskAnalysisResponse(BaseModel):
    risks: str
    negative_sources: List[Dict[str, Any]]
    query: str


class ViewpointComparisonResponse(BaseModel):
    topic: str
    supporting_views: str
    supporting_sources: List[Dict[str, Any]]
    opposing_views: str
    opposing_sources: List[Dict[str, Any]]


class StrategyRequest(BaseModel):
    context: str
    role: Optional[str] = "政策制定者"


class StrategyResponse(BaseModel):
    role: str
    context: str
    strategy: str
    sources: List[Dict[str, Any]]


class SentimentTrendPoint(BaseModel):
    date: str
    sentiment_balance: float
    avg_sentiment_score: float


class SentimentTrendResponse(BaseModel):
    trend: List[SentimentTrendPoint]
