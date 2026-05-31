import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_sample_data(output_path: str, num_comments: int = 500):
    random.seed(20260531)
    np.random.seed(20260531)

    brands = ["NewBrand", "竞品A", "竞品B", "竞品C"]
    products = ["光感修护精华液", "屏障修护面霜", "早C晚A精华套装", "温和洁面乳", "舒缓保湿面膜"]
    platforms = ["小红书", "抖音", "微博", "天猫评价", "京东评价", "B站"]
    campaigns = ["618大促", "新品上市", "达人测评", "敏感肌种草", "夏季修护专题"]
    source_types = ["用户评论", "达人笔记", "电商评价", "短视频评论", "话题讨论"]
    skin_types = ["干敏皮", "油敏皮", "混油皮", "沙漠干皮", "痘肌", "屏障受损肌", "正常偏干", "外油内干"]
    usage_days = ["用了2次", "连续用了3天", "用了1周", "空瓶半瓶后", "刚上脸10分钟", "早晚用了4天"]
    channels = ["直播间入手", "旗舰店买的", "达人链接下单", "朋友推荐后购买", "大促囤货", "小样试用后入正装"]
    detail_notes = [
        "鼻翼和脸颊反应最明显",
        "后续上妆状态变化很大",
        "和同价位竞品对比差异明显",
        "评论区反馈分化很明显",
        "希望品牌补充适用人群说明",
        "客服回复会直接影响我是否复购",
        "这个点对敏感肌很关键",
        "大促期间看到很多类似反馈",
    ]

    positive_templates = [
        "这次{product}真的好用，质地清爽不黏，早上上妆也不搓泥，已经准备回购。",
        "敏感肌用{product}很安心，泛红稳定下来，保湿感也不错。",
        "{brand}这波新品包装和肤感都比以前细腻，吸收快，味道也克制。",
        "连续用了一周，{product}对换季干燥有改善，修护感很明显。",
        "客服解释成分很专业，售后响应也快，这点比竞品体验好。",
        "活动价还可以，套装搭配合理，适合第一次尝试这个系列。",
        "达人测评没有夸张，实际使用确实温和，油皮也能接受。",
        "保湿和提亮都比较自然，不是那种立刻假白的效果。",
    ]

    negative_templates = [
        "用了{product}之后脸颊刺痛泛红，严重辣脸，第二天还起了小疹子。",
        "{brand}广告说温和修护，但我用完明显过敏，这种宣传太离谱。",
        "质地看着高级，上脸却搓泥，后续上妆完全不服帖，踩雷。",
        "价格比竞品贵一截，但效果没有宣传那么夸张，感觉被种草文骗了。",
        "客服一直让我再观察，没有给明确处理方案，售后体验真的差。",
        "买的套装里小样快过期，包装还压坏了，活动体验很糟糕。",
        "用了三天爆痘，停用后才慢慢好，不敢继续用了。",
        "达人笔记全在夸，评论区却很多人说刺激，品牌是不是控评了。",
    ]

    neutral_templates = [
        "{product}肤感还可以，但功效需要再观察，暂时没有特别惊喜。",
        "看到很多人反馈刺痛，也有人说好用，感觉和肤质关系很大。",
        "{brand}这次活动力度一般，赠品多但主品价格没有太大优势。",
        "成分表看起来偏修护，敏感肌最好先做局部测试。",
        "竞品最近也在推类似概念，想看更真实的横评对比。",
        "包装设计挺好看，但更关心长期使用后的稳定性。",
        "客服回复速度还行，但对过敏问题的解释比较模板化。",
        "如果后续能补充第三方检测和适用人群说明，会更放心。",
    ]

    incident_comments = [
        {
            "content": "小红书用户 @美妆纠错本：NewBrand光感修护精华液用了两次就过敏辣脸，脸颊刺痛泛红，客服还让我继续观察，真的离谱。",
            "platform": "小红书",
            "like_count": 1850,
            "comment_count": 426,
            "risk_label": "产品质量风险",
            "campaign": "618大促",
            "source_type": "达人笔记",
        },
        {
            "content": "看到好几个姐妹说NewBrand这支精华液辣脸，我也是同样刺痛，广告里一直讲温和修护，这算不算虚假宣传？",
            "platform": "小红书",
            "like_count": 1240,
            "comment_count": 318,
            "risk_label": "广告承诺争议",
            "campaign": "618大促",
            "source_type": "用户评论",
        },
        {
            "content": "本来冲着敏感肌可用买的，结果用完泛红爆痘，618还在大推这个精华，建议品牌先把问题解释清楚。",
            "platform": "微博",
            "like_count": 920,
            "comment_count": 214,
            "risk_label": "大促风控",
            "campaign": "618大促",
            "source_type": "话题讨论",
        },
    ]

    comments = []
    base_time = datetime(2026, 5, 24, 9, 0, 0)

    for i, item in enumerate(incident_comments):
        comments.append({
            "content": item["content"],
            "publish_time": (datetime(2026, 5, 31, 9, 10, 0) + timedelta(minutes=i * 18)).strftime("%Y-%m-%d %H:%M:%S"),
            "like_count": item["like_count"],
            "comment_count": item["comment_count"],
            "platform": item["platform"],
            "brand": "NewBrand",
            "product": "光感修护精华液",
            "campaign": item["campaign"],
            "source_type": item["source_type"],
            "risk_label": item["risk_label"],
        })

    remaining = max(num_comments - len(comments), 0)
    for _ in range(remaining):
        sentiment = random.choices(["positive", "negative", "neutral"], weights=[0.34, 0.42, 0.24])[0]
        brand = random.choices(brands, weights=[0.58, 0.18, 0.14, 0.10])[0]
        product = random.choice(products)
        campaign = random.choice(campaigns)
        platform = random.choice(platforms)

        if sentiment == "positive":
            template = random.choice(positive_templates)
            risk_label = "正向口碑"
            like_count = int(np.random.gamma(shape=2.2, scale=28))
            comment_count = int(np.random.gamma(shape=1.4, scale=8))
        elif sentiment == "negative":
            template = random.choice(negative_templates)
            risk_label = random.choice(["产品质量风险", "广告承诺争议", "售后体验不满", "价格价值争议", "达人内容质疑"])
            like_count = int(np.random.gamma(shape=2.8, scale=42))
            comment_count = int(np.random.gamma(shape=1.8, scale=13))
        else:
            template = random.choice(neutral_templates)
            risk_label = "中性观察"
            like_count = int(np.random.gamma(shape=1.8, scale=22))
            comment_count = int(np.random.gamma(shape=1.2, scale=6))

        prefix = random.choice(["", "说实话，", "姐妹们，", "测评后感觉，", "用了几天，"])
        suffix = random.choice(["", "大家怎么看？", "希望品牌给个明确说明。", "还会继续观察。", "不建议盲买。"])
        usage_detail = f"我是{random.choice(skin_types)}，{random.choice(channels)}，{random.choice(usage_days)}，{random.choice(detail_notes)}。"
        content = f"{prefix}{template.format(brand=brand, product=product)}{suffix}{usage_detail}"

        publish_time = base_time + timedelta(
            days=random.randint(0, 7),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )

        comments.append({
            "content": content,
            "publish_time": publish_time.strftime("%Y-%m-%d %H:%M:%S"),
            "like_count": like_count,
            "comment_count": comment_count,
            "platform": platform,
            "brand": brand,
            "product": product,
            "campaign": campaign,
            "source_type": random.choice(source_types),
            "risk_label": risk_label,
        })

    df = pd.DataFrame(comments)
    df = df.sample(frac=1, random_state=20260531).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"生成 {len(df)} 条新消费护肤/美妆评论数据，保存至: {output_path}")
    return df


if __name__ == "__main__":
    generate_sample_data("./data/comments.csv", num_comments=500)
