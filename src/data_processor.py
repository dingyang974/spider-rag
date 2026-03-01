import pandas as pd
import re
from typing import List, Dict, Optional
from loguru import logger
from datetime import datetime


class DataProcessor:
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
    
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            self.df = pd.read_csv(self.data_path, encoding='gb18030', on_bad_lines='skip', encoding_errors='replace')
            logger.info(f"Loaded {len(self.df)} records with encoding: gb18030")
            return self.df
        except Exception as e:
            logger.error(f"Error loading with gb18030: {e}")
        
        try:
            self.df = pd.read_csv(self.data_path, encoding='gbk', on_bad_lines='skip', encoding_errors='replace')
            logger.info(f"Loaded {len(self.df)} records with encoding: gbk")
            return self.df
        except Exception as e:
            logger.error(f"Error loading with gbk: {e}")
        
        raise ValueError(f"无法读取文件 {self.data_path}")
    
    def clean_text(self, text: str) -> str:
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'@[\w\u4e00-\u9fff]+[:\s]?', '', text)
        text = re.sub(r'#[\w\u4e00-\u9fff]+#', '', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = text.strip()
        
        return text
    
    def remove_duplicates(self, df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
        if subset is None:
            subset = ['content']
        
        before_count = len(df)
        df = df.drop_duplicates(subset=subset, keep='first')
        after_count = len(df)
        logger.info(f"Removed {before_count - after_count} duplicates")
        return df.reset_index(drop=True)
    
    def split_long_text(self, text: str, max_length: int = 500) -> List[str]:
        if len(text) <= max_length:
            return [text]
        
        sentences = re.split(r'([。！？\.\!\?])', text)
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text[:max_length]]
    
    def normalize_datetime(self, df: pd.DataFrame, time_column: str = 'publish_time') -> pd.DataFrame:
        if time_column not in df.columns:
            logger.warning(f"Column {time_column} not found")
            return df
        
        def parse_time(time_str):
            if pd.isna(time_str):
                return None
            
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y/%m/%d %H:%M:%S",
                "%Y/%m/%d %H:%M",
                "%Y年%m月%d日 %H:%M",
                "%Y年%m月%d日",
                "%Y-%m-%d",
            ]
            
            for fmt in formats:
                try:
                    return pd.to_datetime(time_str, format=fmt)
                except:
                    continue
            
            try:
                return pd.to_datetime(time_str)
            except:
                return None
        
        df[time_column] = df[time_column].apply(parse_time)
        return df
    
    def process(self, 
                remove_duplicates: bool = True,
                clean_text: bool = True,
                normalize_time: bool = True) -> pd.DataFrame:
        
        if self.df is None:
            self.load_data()
        
        df = self.df.copy()
        
        if 'content' not in df.columns:
            raise ValueError("Column 'content' is required. Available columns: " + ", ".join(df.columns))
        
        if clean_text:
            logger.info("Cleaning text...")
            df['content'] = df['content'].apply(self.clean_text)
            df = df[df['content'].str.len() > 0]
        
        if remove_duplicates:
            logger.info("Removing duplicates...")
            df = self.remove_duplicates(df)
        
        if normalize_time and 'publish_time' in df.columns:
            logger.info("Normalizing datetime...")
            df = self.normalize_datetime(df)
        
        for col in ['like_count', 'comment_count']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        df = df.reset_index(drop=True)
        self.df = df
        
        logger.info(f"Processed data: {len(df)} records")
        return df
    
    def get_statistics(self) -> Dict:
        if self.df is None:
            return {}
        
        stats = {
            "total_count": len(self.df),
            "columns": list(self.df.columns),
        }
        
        if 'publish_time' in self.df.columns:
            stats["time_range"] = {
                "start": str(self.df['publish_time'].min()),
                "end": str(self.df['publish_time'].max())
            }
        
        if 'like_count' in self.df.columns:
            stats["like_stats"] = {
                "mean": float(self.df['like_count'].mean()),
                "max": int(self.df['like_count'].max()),
                "total": int(self.df['like_count'].sum())
            }
        
        return stats
