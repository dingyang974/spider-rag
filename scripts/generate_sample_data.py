import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_sample_data(output_path: str, num_comments: int = 500):
    
    positive_templates = [
        "支持这个政策，对年轻人很有帮助",
        "终于有好的政策了，点赞！",
        "这个措施很贴心，希望能落实到位",
        "减轻了很多家庭的负担，支持！",
        "政策很好，但关键是要落实",
        "这是一个进步，期待更多好政策",
        "为这个政策点赞，很人性化",
        "终于有人关心这个问题了",
        "这个政策很及时，解决了实际问题",
        "支持支持！希望能推广开来",
    ]
    
    negative_templates = [
        "这个政策根本解决不了问题",
        "形式主义，没有实际意义",
        "负担太重了，年轻人压力很大",
        "房价那么高，这点补贴有什么用",
        "教育成本太高，养不起孩子",
        "工作太忙，没时间照顾孩子",
        "托育机构太少，没人帮忙带孩子",
        "这个政策脱离实际",
        "又是画大饼，实际效果呢？",
        "年轻人工资太低，怎么敢生孩子",
    ]
    
    neutral_templates = [
        "政策出台了，关键看执行",
        "希望能有更多配套措施",
        "这个政策的具体内容是什么",
        "还需要观察实际效果",
        "政策方向是对的，但细节很重要",
        "不同地区情况不同，需要因地制宜",
        "期待看到更多相关报道",
        "这个话题值得深入讨论",
        "希望能听到更多人的声音",
        "政策解读很重要",
    ]
    
    topics = [
        "生育补贴", "托育服务", "产假政策", "教育成本", 
        "住房问题", "工作平衡", "医疗保障", "育儿支持"
    ]
    
    comments = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(num_comments):
        sentiment = random.choices(
            ["positive", "negative", "neutral"],
            weights=[0.3, 0.4, 0.3]
        )[0]
        
        if sentiment == "positive":
            content = random.choice(positive_templates)
        elif sentiment == "negative":
            content = random.choice(negative_templates)
        else:
            content = random.choice(neutral_templates)
        
        variations = [
            content,
            content + "，大家怎么看？",
            "我觉得" + content,
            content + "！",
            content + "。",
            "说实话，" + content,
        ]
        content = random.choice(variations)
        
        topic = random.choice(topics)
        content = f"关于{topic}，" + content
        
        days_offset = random.randint(0, 90)
        hours_offset = random.randint(0, 23)
        minutes_offset = random.randint(0, 59)
        publish_time = base_date + timedelta(
            days=days_offset,
            hours=hours_offset,
            minutes=minutes_offset
        )
        
        like_count = int(np.random.exponential(scale=20))
        comment_count = int(np.random.exponential(scale=5))
        
        comments.append({
            "content": content,
            "publish_time": publish_time.strftime("%Y-%m-%d %H:%M:%S"),
            "like_count": like_count,
            "comment_count": comment_count,
        })
    
    df = pd.DataFrame(comments)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"生成 {len(df)} 条示例评论数据，保存至: {output_path}")
    return df


if __name__ == "__main__":
    generate_sample_data("./data/comments.csv", num_comments=500)
