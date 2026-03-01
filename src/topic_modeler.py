import pandas as pd
from typing import List, Dict, Tuple, Optional
from loguru import logger
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import numpy as np


class TopicModeler:
    
    def __init__(self, n_topics: int = 5, n_words: int = 10):
        self.n_topics = n_topics
        self.n_words = n_words
        self.lda_model: Optional[LatentDirichletAllocation] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.topic_words: List[List[str]] = []
    
    def tokenize(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        
        stop_words = {
            '的', '了', '是', '在', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '什么',
            '他', '她', '它', '这个', '那个', '这些', '那些', '可以', '没',
            '但是', '因为', '所以', '如果', '虽然', '还是', '或者', '而且',
            '就是', '这样', '那样', '怎么', '为什么', '哪', '哪里', '哪个'
        }
        
        words = jieba.cut(text)
        words = [w for w in words if w.strip() and w not in stop_words and len(w) > 1]
        return ' '.join(words)
    
    def fit(self, df: pd.DataFrame, text_column: str = 'content') -> 'TopicModeler':
        logger.info(f"Fitting topic model on {len(df)} documents...")
        
        texts = df[text_column].apply(self.tokenize).tolist()
        texts = [t for t in texts if t.strip()]
        
        if not texts:
            logger.warning("No valid texts for topic modeling")
            return self
        
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=10,
            learning_method='online',
            random_state=42,
            n_jobs=-1
        )
        
        self.lda_model.fit(tfidf_matrix)
        
        self._extract_topic_words()
        
        logger.info("Topic model fitting completed")
        return self
    
    def _extract_topic_words(self):
        if self.lda_model is None or self.vectorizer is None:
            return
        
        feature_names = self.vectorizer.get_feature_names_out()
        self.topic_words = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[:-self.n_words - 1:-1]
            top_words = [feature_names[i] for i in top_indices]
            self.topic_words.append(top_words)
    
    def get_topics(self) -> List[Dict]:
        topics = []
        for i, words in enumerate(self.topic_words):
            topics.append({
                "topic_id": i,
                "keywords": words,
                "label": self._generate_topic_label(words)
            })
        return topics
    
    def _generate_topic_label(self, words: List[str]) -> str:
        if not words:
            return "未知主题"
        
        return f"主题: {', '.join(words[:3])}"
    
    def transform(self, df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
        if self.lda_model is None or self.vectorizer is None:
            logger.warning("Model not fitted, please call fit() first")
            return df
        
        df = df.copy()
        texts = df[text_column].apply(self.tokenize).tolist()
        
        tfidf_matrix = self.vectorizer.transform(texts)
        topic_distributions = self.lda_model.transform(tfidf_matrix)
        
        df['topic_distribution'] = list(topic_distributions)
        df['dominant_topic'] = topic_distributions.argmax(axis=1)
        df['topic_confidence'] = topic_distributions.max(axis=1)
        
        return df
    
    def get_topic_distribution(self, df: pd.DataFrame) -> Dict:
        if 'dominant_topic' not in df.columns:
            return {}
        
        distribution = df['dominant_topic'].value_counts().to_dict()
        total = len(df)
        
        result = {}
        for topic_id in range(self.n_topics):
            count = distribution.get(topic_id, 0)
            result[f"topic_{topic_id}"] = {
                "count": count,
                "ratio": round(count / total, 3) if total > 0 else 0,
                "keywords": self.topic_words[topic_id] if topic_id < len(self.topic_words) else []
            }
        
        return result
    
    def extract_keywords(self, df: pd.DataFrame, text_column: str = 'content',
                        top_n: int = 30) -> List[Tuple[str, float]]:
        texts = df[text_column].tolist()
        all_text = ' '.join([str(t) for t in texts if pd.notna(t)])
        
        keywords = jieba.analyse.extract_tags(all_text, topK=top_n, withWeight=True)
        return keywords
    
    def get_topic_trend(self, df: pd.DataFrame, time_column: str = 'publish_time',
                       freq: str = 'D') -> pd.DataFrame:
        if 'dominant_topic' not in df.columns or time_column not in df.columns:
            return pd.DataFrame()
        
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.dropna(subset=[time_column])
        
        topic_counts = df.groupby([
            pd.Grouper(key=time_column, freq=freq),
            'dominant_topic'
        ]).size().unstack(fill_value=0)
        
        return topic_counts.reset_index()
