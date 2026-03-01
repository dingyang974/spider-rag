import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import DataProcessor
from src.sentiment_analyzer import SentimentAnalyzer
from src.topic_modeler import TopicModeler
from src.vector_store import VectorStore
from config import settings
from loguru import logger


def run_pipeline(data_path: str = None, force_rebuild: bool = False):
    
    logger.info("=" * 50)
    logger.info("开始构建舆情分析知识库")
    logger.info("=" * 50)
    
    data_path = data_path or settings.DATA_PATH
    
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        logger.info("正在生成示例数据...")
        from scripts.generate_sample_data import generate_sample_data
        generate_sample_data(data_path, num_comments=500)
    
    logger.info("\n[1/5] 数据预处理...")
    processor = DataProcessor(data_path)
    df = processor.process()
    stats = processor.get_statistics()
    logger.info(f"处理完成: {stats}")
    
    logger.info("\n[2/5] 情感分析...")
    sentiment_analyzer = SentimentAnalyzer()
    df = sentiment_analyzer.analyze_dataframe(df)
    sentiment_dist = sentiment_analyzer.get_sentiment_distribution(df)
    logger.info(f"情感分布: {sentiment_dist}")
    
    logger.info("\n[3/5] 主题建模...")
    topic_modeler = TopicModeler(n_topics=5, n_words=10)
    topic_modeler.fit(df)
    df = topic_modeler.transform(df)
    topics = topic_modeler.get_topics()
    logger.info(f"识别到 {len(topics)} 个主题")
    for topic in topics:
        logger.info(f"  - {topic['label']}: {', '.join(topic['keywords'][:5])}")
    
    logger.info("\n[4/5] 构建向量索引...")
    vector_store = VectorStore()
    vector_store.build_index(df)
    
    logger.info("\n[5/5] 保存知识库...")
    os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
    vector_store.save()
    
    processed_data_path = os.path.join(settings.VECTOR_STORE_PATH, "processed_data.csv")
    df.to_csv(processed_data_path, index=False, encoding='utf-8-sig')
    
    logger.info("\n" + "=" * 50)
    logger.info("知识库构建完成！")
    logger.info(f"总文档数: {len(df)}")
    logger.info(f"向量库路径: {settings.VECTOR_STORE_PATH}")
    logger.info("=" * 50)
    
    return {
        "total_documents": len(df),
        "sentiment_distribution": sentiment_dist,
        "topics": topics,
        "vector_store_stats": vector_store.get_statistics()
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="构建舆情分析知识库")
    parser.add_argument("--data", type=str, help="数据文件路径")
    parser.add_argument("--force", action="store_true", help="强制重建")
    
    args = parser.parse_args()
    
    result = run_pipeline(data_path=args.data, force_rebuild=args.force)
    print("\n构建结果:", result)
