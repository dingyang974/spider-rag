import pandas as pd
from typing import List, Dict, Tuple, Optional
from loguru import logger
import jieba
import jieba.analyse
from collections import Counter


class SentimentAnalyzer:
    
    POSITIVE_WORDS = {
        '支持', '赞同', '同意', '好', '棒', '优秀', '合理', '正确', '应该',
        '希望', '期待', '感谢', '点赞', '认可', '理解', '支持', '不错',
        '很好', '太好', '真好', '真好', '点赞', '给力', '靠谱', '明智',
        '进步', '改善', '利好', '福利', '方便', '实用', '贴心', '温暖'
    }
    
    NEGATIVE_WORDS = {
        '反对', '不赞同', '不同意', '不好', '差', '糟糕', '不合理', '错误',
        '不应该', '失望', '愤怒', '不满', '抱怨', '批评', '质疑', '担心',
        '忧虑', '恐惧', '害怕', '讨厌', '恶心', '无语', '可笑', '荒谬',
        '过分', '离谱', '坑', '骗', '假', '虚伪', '形式主义', '官僚'
    }
    
    INTENSIFIERS = {
        '非常', '很', '太', '特别', '极其', '相当', '十分', '超级',
        '真的', '实在', '确实', '绝对', '一定', '肯定'
    }
    
    NEGATION_WORDS = {
        '不', '没', '无', '非', '未', '别', '莫', '勿', '不是', '没有'
    }
    
    def __init__(self):
        self.sentiment_lexicon = self._build_lexicon()
    
    def _build_lexicon(self) -> Dict[str, int]:
        lexicon = {}
        for word in self.POSITIVE_WORDS:
            lexicon[word] = 1
        for word in self.NEGATIVE_WORDS:
            lexicon[word] = -1
        return lexicon
    
    def analyze_text(self, text: str) -> Dict:
        if not text or not isinstance(text, str):
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}
        
        words = list(jieba.cut(text))
        
        score = 0
        word_count = 0
        negation_flag = False
        intensifier_multiplier = 1.0
        
        for i, word in enumerate(words):
            if word in self.NEGATION_WORDS:
                negation_flag = True
                continue
            
            if word in self.INTENSIFIERS:
                intensifier_multiplier = 1.5
                continue
            
            if word in self.sentiment_lexicon:
                word_score = self.sentiment_lexicon[word]
                
                if negation_flag:
                    word_score *= -1
                    negation_flag = False
                
                word_score *= intensifier_multiplier
                intensifier_multiplier = 1.0
                
                score += word_score
                word_count += 1
        
        if word_count == 0:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.5}
        
        normalized_score = score / max(word_count, 1)
        
        if normalized_score > 0.3:
            sentiment = "positive"
        elif normalized_score < -0.3:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        confidence = min(abs(normalized_score) + 0.3, 1.0)
        
        return {
            "sentiment": sentiment,
            "score": round(normalized_score, 3),
            "confidence": round(confidence, 3)
        }
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
        logger.info(f"Analyzing sentiment for {len(df)} records...")
        
        results = df[text_column].apply(self.analyze_text)
        
        df = df.copy()
        df['sentiment'] = results.apply(lambda x: x['sentiment'])
        df['sentiment_score'] = results.apply(lambda x: x['score'])
        df['sentiment_confidence'] = results.apply(lambda x: x['confidence'])
        
        logger.info(f"Sentiment analysis completed")
        return df
    
    def get_sentiment_distribution(self, df: pd.DataFrame) -> Dict:
        if 'sentiment' not in df.columns:
            return {}
        
        distribution = df['sentiment'].value_counts().to_dict()
        total = len(df)
        
        return {
            "positive": distribution.get('positive', 0),
            "negative": distribution.get('negative', 0),
            "neutral": distribution.get('neutral', 0),
            "positive_ratio": round(distribution.get('positive', 0) / total, 3) if total > 0 else 0,
            "negative_ratio": round(distribution.get('negative', 0) / total, 3) if total > 0 else 0,
            "neutral_ratio": round(distribution.get('neutral', 0) / total, 3) if total > 0 else 0,
        }
    
    def get_sentiment_trend(self, df: pd.DataFrame, time_column: str = 'publish_time', 
                           freq: str = 'D') -> pd.DataFrame:
        if 'sentiment' not in df.columns or time_column not in df.columns:
            return pd.DataFrame()
        
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.dropna(subset=[time_column])
        
        trend = df.groupby(pd.Grouper(key=time_column, freq=freq)).agg({
            'sentiment': lambda x: (x == 'positive').sum() - (x == 'negative').sum(),
            'sentiment_score': 'mean'
        }).reset_index()
        
        trend.columns = ['date', 'sentiment_balance', 'avg_sentiment_score']
        return trend
    
    def get_negative_keywords(self, df: pd.DataFrame, text_column: str = 'content',
                             top_n: int = 20) -> List[Tuple[str, int]]:
        if 'sentiment' not in df.columns:
            return []
        
        negative_texts = df[df['sentiment'] == 'negative'][text_column].tolist()
        negative_text = ' '.join([str(t) for t in negative_texts if pd.notna(t)])
        
        keywords = jieba.analyse.extract_tags(negative_text, topK=top_n, withWeight=True)
        return keywords
