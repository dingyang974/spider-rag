import os
import json
import pickle
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import faiss
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from config import settings


class VectorStore:
    
    def __init__(self, 
                 dimension: int = None,
                 embedding_model: str = None):
        self.dimension = dimension or 768
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[Dict] = []
        logger.info("Using TF-IDF based vector store (no model download required)")
    
    def tokenize_chinese(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        words = jieba.cut(text)
        return " ".join(words)
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        logger.info(f"Encoding {len(texts)} texts with TF-IDF...")
        
        tokenized_texts = [self.tokenize_chinese(text) for text in texts]
        
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=self.dimension,
                min_df=1,
                max_df=0.95,
                ngram_range=(1, 2)
            )
            tfidf_matrix = self.vectorizer.fit_transform(tokenized_texts)
        else:
            tfidf_matrix = self.vectorizer.transform(tokenized_texts)
        
        dense_matrix = tfidf_matrix.toarray().astype(np.float32)
        
        norms = np.linalg.norm(dense_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        dense_matrix = dense_matrix / norms
        
        return dense_matrix
    
    def get_embedding(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str):
            return np.zeros(self.dimension, dtype=np.float32)
        
        tokenized = self.tokenize_chinese(text)
        tfidf_vec = self.vectorizer.transform([tokenized])
        embedding = tfidf_vec.toarray().astype(np.float32)[0]
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
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
        self.dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        logger.info(f"Vector index built with {self.index.ntotal} vectors, dimension: {self.dimension}")
    
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
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
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
        
        with open(os.path.join(path, "vectorizer.pkl"), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(os.path.join(path, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump({
                "dimension": self.dimension,
                "embedding_model": "tfidf",
                "total_documents": len(self.documents)
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Vector store saved to {path}")
    
    def load(self, path: str = None) -> None:
        path = path or settings.VECTOR_STORE_PATH
        
        index_path = os.path.join(path, "index.faiss")
        docs_path = os.path.join(path, "documents.pkl")
        vectorizer_path = os.path.join(path, "vectorizer.pkl")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(index_path)
            self.dimension = self.index.d
            
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
            
            logger.info(f"Vector store loaded: {len(self.documents)} documents")
        else:
            logger.warning(f"Vector store files not found at {path}")
    
    def get_statistics(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "embedding_model": "tfidf",
            "index_built": self.index is not None
        }
