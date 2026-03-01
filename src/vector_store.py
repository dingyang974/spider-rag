import os
import json
import pickle
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import faiss
from loguru import logger
from config import settings

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer


class VectorStore:
    
    def __init__(self, 
                 dimension: int = None,
                 embedding_model: str = None):
        self.embedding_model_name = embedding_model or "shibing624/text2vec-base-chinese"
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        logger.info("Using HF mirror: https://hf-mirror.com")
        self.model = SentenceTransformer(self.embedding_model_name)
        self.dimension = dimension or self.model.get_sentence_embedding_dimension()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[Dict] = []
    
    def get_embedding(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str):
            return np.zeros(self.dimension, dtype=np.float32)
        
        try:
            embedding = self.model.encode(text[:512], convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return np.zeros(self.dimension, dtype=np.float32)
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        logger.info(f"Encoding {len(texts)} texts with local model...")
        
        valid_texts = []
        for text in texts:
            if text and isinstance(text, str):
                valid_texts.append(text[:512])
            else:
                valid_texts.append("")
        
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings.astype(np.float32)
    
    def build_index(self, df: pd.DataFrame, text_column: str = 'content') -> None:
        logger.info(f"Building vector index for {len(df)} documents...")
        
        self.documents = []
        texts = []
        
        for idx, row in df.iterrows():
            doc = {
                "id": idx,
                "content": str(row.get(text_column, '')),
                "sentiment": row.get('sentiment', 'neutral'),
                "sentiment_score": row.get('sentiment_score', 0),
                "topic": row.get('dominant_topic', -1),
                "publish_time": str(row.get('publish_time', '')),
                "like_count": int(row.get('like_count', 0)),
                "comment_count": int(row.get('comment_count', 0))
            }
            self.documents.append(doc)
            texts.append(doc["content"])
        
        embeddings = self.get_embeddings_batch(texts)
        
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.nan_to_num(embeddings)
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        logger.info(f"Vector index built with {self.index.ntotal} vectors")
    
    def search(self, 
               query: str, 
               top_k: int = None,
               sentiment_filter: Optional[str] = None,
               min_likes: int = 0) -> List[Dict]:
        
        if self.index is None:
            logger.warning("Index not built")
            return []
        
        top_k = top_k or settings.TOP_K_RETRIEVAL
        
        query_embedding = self.get_embedding(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = self.index.search(query_embedding, min(top_k * 3, len(self.documents)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.documents):
                continue
            
            doc = self.documents[idx].copy()
            doc['similarity_score'] = float(score)
            
            if sentiment_filter and doc['sentiment'] != sentiment_filter:
                continue
            
            if doc['like_count'] < min_likes:
                continue
            
            results.append(doc)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def save(self, path: str = None) -> None:
        path = path or settings.VECTOR_STORE_PATH
        os.makedirs(path, exist_ok=True)
        
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        with open(os.path.join(path, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
        
        with open(os.path.join(path, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump({
                "dimension": self.dimension,
                "embedding_model": self.embedding_model_name,
                "total_documents": len(self.documents)
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Vector store saved to {path}")
    
    def load(self, path: str = None) -> None:
        path = path or settings.VECTOR_STORE_PATH
        
        index_path = os.path.join(path, "index.faiss")
        docs_path = os.path.join(path, "documents.pkl")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(index_path)
            
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            logger.info(f"Vector store loaded: {len(self.documents)} documents")
        else:
            logger.warning(f"Vector store files not found at {path}")
    
    def get_statistics(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "embedding_model": self.embedding_model_name,
            "index_built": self.index is not None
        }
