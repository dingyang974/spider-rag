from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import pandas as pd
from loguru import logger
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import DataProcessor
from src.sentiment_analyzer import SentimentAnalyzer
from src.topic_modeler import TopicModeler
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from config import settings
from api.schemas import (
    QueryRequest, QueryResponse,
    OverviewResponse, SentimentDistribution, TopicInfo,
    CommentsResponse, CommentItem,
    BuildKnowledgeBaseRequest, BuildKnowledgeBaseResponse,
    RiskAnalysisResponse, ViewpointComparisonResponse,
    StrategyRequest, StrategyResponse,
    SentimentTrendResponse, SentimentTrendPoint
)

global_state = {
    "df": None,
    "vector_store": None,
    "rag_engine": None,
    "sentiment_analyzer": None,
    "topic_modeler": None,
    "is_initialized": False
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing application...")
    
    try:
        if os.path.exists(settings.VECTOR_STORE_PATH):
            global_state["vector_store"] = VectorStore()
            global_state["vector_store"].load()
            global_state["rag_engine"] = RAGEngine(global_state["vector_store"])
            global_state["is_initialized"] = True
            logger.info("Loaded existing vector store")
    except Exception as e:
        logger.warning(f"Could not load existing vector store: {e}")
    
    yield
    
    logger.info("Shutting down application...")


app = FastAPI(
    title="生育议题舆情智能分析与决策助手",
    description="基于RAG的舆情智能问答系统API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_rag_engine() -> RAGEngine:
    if not global_state["is_initialized"]:
        raise HTTPException(status_code=503, detail="知识库未初始化，请先构建知识库")
    return global_state["rag_engine"]


def get_dataframe() -> pd.DataFrame:
    if global_state["df"] is None:
        raise HTTPException(status_code=503, detail="数据未加载")
    return global_state["df"]


@app.get("/")
async def root():
    return {
        "message": "生育议题舆情智能分析与决策助手 API",
        "version": "1.0.0",
        "status": "initialized" if global_state["is_initialized"] else "not_initialized"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_store": global_state["vector_store"] is not None,
        "rag_engine": global_state["rag_engine"] is not None,
        "data_loaded": global_state["df"] is not None
    }


@app.post("/api/build-knowledge-base", response_model=BuildKnowledgeBaseResponse)
async def build_knowledge_base(request: BuildKnowledgeBaseRequest):
    logger.info("Building knowledge base...")
    
    try:
        data_path = request.data_path or settings.DATA_PATH
        
        if not os.path.exists(data_path):
            raise HTTPException(status_code=400, detail=f"数据文件不存在: {data_path}")
        
        data_processor = DataProcessor(data_path)
        df = data_processor.process()
        
        sentiment_analyzer = SentimentAnalyzer()
        df = sentiment_analyzer.analyze_dataframe(df)
        global_state["sentiment_analyzer"] = sentiment_analyzer
        
        topic_modeler = TopicModeler(n_topics=5, n_words=10)
        topic_modeler.fit(df)
        df = topic_modeler.transform(df)
        global_state["topic_modeler"] = topic_modeler
        
        vector_store = VectorStore()
        vector_store.build_index(df)
        vector_store.save()
        
        global_state["df"] = df
        global_state["vector_store"] = vector_store
        global_state["rag_engine"] = RAGEngine(vector_store)
        global_state["is_initialized"] = True
        
        return BuildKnowledgeBaseResponse(
            success=True,
            message="知识库构建成功",
            documents_count=len(df)
        )
        
    except Exception as e:
        logger.error(f"Error building knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/overview", response_model=OverviewResponse)
async def get_overview(df: pd.DataFrame = Depends(get_dataframe)):
    sentiment_analyzer = global_state.get("sentiment_analyzer")
    topic_modeler = global_state.get("topic_modeler")
    
    sentiment_dist = {"positive": 0, "negative": 0, "neutral": 0}
    if sentiment_analyzer and 'sentiment' in df.columns:
        sentiment_dist = sentiment_analyzer.get_sentiment_distribution(df)
    
    topics = []
    if topic_modeler:
        topics = [TopicInfo(**t) for t in topic_modeler.get_topics()]
    
    time_range = None
    if 'publish_time' in df.columns:
        time_range = {
            "start": str(df['publish_time'].min()),
            "end": str(df['publish_time'].max())
        }
    
    top_keywords = []
    if topic_modeler:
        keywords = topic_modeler.extract_keywords(df, top_n=20)
        top_keywords = [{"word": k, "weight": float(w)} for k, w in keywords]
    
    return OverviewResponse(
        total_comments=len(df),
        sentiment_distribution=SentimentDistribution(**sentiment_dist),
        topics=topics,
        time_range=time_range,
        top_keywords=top_keywords
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest, rag_engine: RAGEngine = Depends(get_rag_engine)):
    try:
        result = rag_engine.query(
            question=request.question,
            top_k=request.top_k,
            sentiment_filter=request.sentiment_filter,
            min_likes=request.min_likes,
            conversation_history=request.conversation_history
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Error in query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/comments", response_model=CommentsResponse)
async def get_comments(
    page: int = 1,
    page_size: int = 20,
    sentiment: Optional[str] = None,
    sort_by: Optional[str] = "like_count",
    df: pd.DataFrame = Depends(get_dataframe)
):
    filtered_df = df.copy()
    
    if sentiment and 'sentiment' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['sentiment'] == sentiment]
    
    if sort_by and sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)
    
    total = len(filtered_df)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    page_df = filtered_df.iloc[start_idx:end_idx]
    
    comments = []
    for idx, row in page_df.iterrows():
        comments.append(CommentItem(
            id=int(idx),
            content=str(row.get('content', '')),
            sentiment=str(row.get('sentiment', 'neutral')),
            sentiment_score=float(row.get('sentiment_score', 0)),
            like_count=int(row.get('like_count', 0)),
            publish_time=str(row.get('publish_time', ''))
        ))
    
    return CommentsResponse(
        comments=comments,
        total=total,
        page=page,
        page_size=page_size
    )


@app.get("/api/sentiment-trend", response_model=SentimentTrendResponse)
async def get_sentiment_trend(
    freq: str = "D",
    df: pd.DataFrame = Depends(get_dataframe)
):
    sentiment_analyzer = global_state.get("sentiment_analyzer")
    
    if not sentiment_analyzer:
        raise HTTPException(status_code=503, detail="情感分析器未初始化")
    
    trend_df = sentiment_analyzer.get_sentiment_trend(df, freq=freq)
    
    trend = []
    for _, row in trend_df.iterrows():
        trend.append(SentimentTrendPoint(
            date=str(row['date']),
            sentiment_balance=float(row['sentiment_balance']),
            avg_sentiment_score=float(row['avg_sentiment_score'])
        ))
    
    return SentimentTrendResponse(trend=trend)


@app.get("/api/risks", response_model=RiskAnalysisResponse)
async def identify_risks(
    topic: Optional[str] = None,
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    try:
        result = rag_engine.identify_risks(topic=topic)
        return RiskAnalysisResponse(**result)
    except Exception as e:
        logger.error(f"Error identifying risks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/viewpoints/{topic}", response_model=ViewpointComparisonResponse)
async def compare_viewpoints(
    topic: str,
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    try:
        result = rag_engine.compare_viewpoints(topic=topic)
        return ViewpointComparisonResponse(**result)
    except Exception as e:
        logger.error(f"Error comparing viewpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/strategy", response_model=StrategyResponse)
async def generate_strategy(
    request: StrategyRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    try:
        result = rag_engine.generate_strategy(
            context=request.context,
            role=request.role
        )
        return StrategyResponse(**result)
    except Exception as e:
        logger.error(f"Error generating strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/statistics")
async def get_statistics(df: pd.DataFrame = Depends(get_dataframe)):
    stats = {
        "total_comments": len(df),
        "columns": list(df.columns),
    }
    
    if 'sentiment' in df.columns:
        stats["sentiment_counts"] = df['sentiment'].value_counts().to_dict()
    
    if 'like_count' in df.columns:
        stats["like_statistics"] = {
            "mean": float(df['like_count'].mean()),
            "max": int(df['like_count'].max()),
            "min": int(df['like_count'].min()),
            "sum": int(df['like_count'].sum())
        }
    
    vector_store = global_state.get("vector_store")
    if vector_store:
        stats["vector_store"] = vector_store.get_statistics()
    
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
