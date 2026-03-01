import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from config import settings
from loguru import logger


def test_rag():
    
    logger.info("测试 RAG 系统...")
    
    vector_store = VectorStore()
    vector_store.load()
    
    rag_engine = RAGEngine(vector_store)
    
    test_queries = [
        "最近讨论焦点是什么？",
        "当前负面情绪主要集中在哪些方面？",
        "支持与反对的核心观点分别是什么？",
    ]
    
    for query in test_queries:
        logger.info(f"\n问题: {query}")
        logger.info("-" * 50)
        
        result = rag_engine.query(query, top_k=5)
        
        print(f"\n回答:\n{result['answer']}")
        print(f"\n检索到 {result['retrieval_count']} 条相关评论")
        
        if result['sources']:
            print("\n相关评论示例:")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"  {i}. {source['content'][:50]}... (相似度: {source['similarity_score']:.3f})")


if __name__ == "__main__":
    test_rag()
