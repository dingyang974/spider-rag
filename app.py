import os
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

DEFAULT_API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="InsightOps 企业市场情报 Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_runtime_setting(key: str, default: str) -> str:
    env_value = os.getenv(key)
    if env_value:
        return env_value

    secrets_paths = [
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ]
    if not any(os.path.exists(path) for path in secrets_paths):
        return default

    try:
        value = st.secrets.get(key)
    except Exception:
        value = None
    return value or default


API_BASE_URL = get_runtime_setting("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


DEFAULT_WORK_VIEW = "品牌运营"


ROLE_VIEWS: Dict[str, Dict] = {
    "品牌运营": {
        "subtitle": "统筹品牌声量、监测项目、跨团队处置和情报交付。",
        "metrics": [
            ("全网声量", "124,800", "+18.4%", "up"),
            ("待研判信号", "18", "+5", "warn"),
            ("跨团队事项", "6", "3 项需分派", "warn"),
            ("本周报告", "6", "已交付 4", "ok"),
        ],
        "focus": ["确认广告争议是否升级", "查看竞品 A 新品声量", "安排今日市场情报日报"],
        "agent_queue": [
            ("广告争议是否升级为跨团队事件", "待确认负责人", "R-024"),
            ("竞品 A 新品声量异常", "证据汇总中", "C-118"),
            ("售后响应问题是否影响品牌口碑", "待人工确认", "R-021"),
            ("春季上新活动复盘结论", "报告生成中", "M-047"),
        ],
        "agent_output": "市场情报日报、跨团队分派清单、今日重点观察项",
        "risk_lens": "品牌运营视角优先判断影响面、负责人和升级机制，重点看事件是否需要跨团队协同处置。",
        "risk_actions": ["分派负责人", "加入今日风险简报", "创建跨团队跟进", "追踪 48 小时扩散"],
        "competitor_focus": "关注竞品变化是否影响本品牌声量结构，并把风险、机会和待跟进事项同步到日报。",
        "report_defaults": {
            "type": "市场情报日报",
            "audience": "品牌运营负责人",
            "modules": ["关键结论", "风险事件", "竞品变化", "建议动作"],
        },
        "copilot_commands": [
            {
                "button": "生成日报",
                "title": "生成日报",
                "desc": "汇总今日市场情报",
                "icon": "D",
                "answer": "已生成市场情报日报草稿：今日重点为广告争议风险、竞品 A 新品声量上升和售后响应讨论。建议品牌运营先分派风险负责人，再把竞品观察同步给竞品策略团队。",
            },
            {
                "button": "分派任务",
                "title": "任务分派",
                "desc": "把信号转为负责人动作",
                "icon": "T",
                "answer": "建议分派：公关风控负责广告争议回应口径，营销增长复盘素材承诺表达，竞品策略补充竞品 A 对比证据，品牌运营在今日日报中跟踪闭环状态。",
            },
            {
                "button": "整理重点",
                "title": "重点提炼",
                "desc": "提炼三条管理摘要",
                "icon": "S",
                "answer": "今日三条重点：一是广告文案争议已形成高赞负评，二是竞品 A 新品带动价格对比，三是售后响应慢仍在积累负面体验。建议优先处理第一项。",
            },
            {
                "button": "加入简报",
                "title": "报告沉淀",
                "desc": "写入今日情报简报",
                "icon": "R",
                "answer": "已生成简报段落：今日品牌口碑主要受广告承诺和价格感知影响，需在 48 小时内观察负面扩散，并同步竞品对比讨论给竞品策略团队。",
            },
        ],
    },
    "公关风控": {
        "subtitle": "负责风险事件、高赞负评、证据链、回应口径和升级判断。",
        "metrics": [
            ("高危舆情", "3", "需 2 小时内处理", "bad"),
            ("高赞负评", "24", "+9", "warn"),
            ("核心平台", "小红书", "负面集中", "bad"),
            ("回应口径", "5", "待审核 2", "ok"),
        ],
        "focus": ["处理价格争议事件", "审核客服统一回复", "追踪中腰部 KOL 扩散"],
        "agent_queue": [
            ("广告争议归因与事实边界", "已完成", "R-024"),
            ("高赞负评证据链整理", "证据汇总中", "E-203"),
            ("中腰部 KOL 二次扩散判断", "待人工确认", "R-029"),
            ("客服回应口径一致性检查", "待审核", "S-014"),
        ],
        "agent_output": "回应口径、证据摘要、升级建议、客服 FAQ 更新",
        "risk_lens": "公关风控视角优先判断事实边界、传播速度和回应窗口，重点看是否会从用户吐槽升级为媒体议题。",
        "risk_actions": ["生成回应口径", "整理证据摘要", "标记高赞负评", "同步客服 FAQ"],
        "competitor_focus": "关注竞品对比是否加剧负面情绪，优先提取可公开回应的事实点，避免陷入无依据对比。",
        "report_defaults": {
            "type": "风险事件简报",
            "audience": "公关风控负责人",
            "modules": ["风险事件", "证据评论", "建议动作", "下周观察点"],
        },
        "copilot_commands": [
            {
                "button": "生成回应口径",
                "title": "回应口径",
                "desc": "生成对外回应草稿",
                "icon": "P",
                "answer": "建议回应口径：感谢用户反馈，我们已关注到大家对新品价格和实际体验的讨论。新品定价综合考虑研发、服务和长期使用成本，后续会补充更清晰的功能说明与真实使用场景，也欢迎继续反馈具体体验。",
            },
            {
                "button": "解释风险原因",
                "title": "风险归因",
                "desc": "解释风险为什么升高",
                "icon": "R",
                "answer": "该事件被判定为高风险，核心原因是负面增长速度快、高赞评论集中，且讨论已经从价格吐槽扩散到广告承诺与竞品对比。建议先确认事实性表述是否过强，再统一公关与客服口径。",
            },
            {
                "button": "整理证据摘要",
                "title": "证据整理",
                "desc": "汇总关键评论来源",
                "icon": "E",
                "answer": "证据摘要：当前共识别 126 条相关评论，其中高赞负评 18 条。主要集中在小红书和微博，关键词包括价格虚高、广告说太满、竞品更划算。传播路径暂未进入媒体报道，但已出现中腰部账号二次讨论。",
            },
            {
                "button": "生成客服 FAQ",
                "title": "客服 FAQ",
                "desc": "转为一线回应材料",
                "icon": "F",
                "answer": "客服 FAQ 建议补充三类问题：新品定价逻辑、广告承诺的适用条件、与竞品对比时的真实差异。回复中避免绝对化承诺，并引导用户描述具体使用场景。",
            },
        ],
    },
    "营销增长": {
        "subtitle": "负责活动反馈、广告素材、用户反感点、投放优化和复盘沉淀。",
        "metrics": [
            ("活动提及", "38,420", "+26.1%", "up"),
            ("素材争议点", "7", "+3", "warn"),
            ("正向卖点", "12", "可复用", "ok"),
            ("待复盘活动", "2", "本周截止", "warn"),
        ],
        "focus": ["复盘春季上新 campaign", "定位短视频评论反感点", "生成素材优化建议"],
        "agent_queue": [
            ("春季上新活动评论复盘", "报告生成中", "M-047"),
            ("短视频素材反感点聚类", "证据汇总中", "A-066"),
            ("正向卖点可复用清单", "待人工确认", "M-052"),
            ("达人笔记转化反馈对比", "分析中", "K-031"),
        ],
        "agent_output": "活动复盘、素材优化建议、用户反感点清单、下一轮投放假设",
        "risk_lens": "营销增长视角优先判断争议是否来自素材表达、卖点承诺或投放人群错配，重点反推下一轮投放如何改。",
        "risk_actions": ["定位反感点", "生成素材优化建议", "标记不可复用表达", "加入活动复盘"],
        "competitor_focus": "关注竞品素材、达人测评和促销表达中哪些内容带来有效声量，可转化为下一轮投放假设。",
        "report_defaults": {
            "type": "营销活动复盘",
            "audience": "营销增长负责人",
            "modules": ["关键结论", "证据评论", "建议动作", "下周观察点"],
        },
        "copilot_commands": [
            {
                "button": "生成活动复盘",
                "title": "活动复盘",
                "desc": "沉淀投放表现与反馈",
                "icon": "M",
                "answer": "活动复盘草稿：春季上新 campaign 带来明显声量增长，但争议集中在广告承诺表达过满和价格感知落差。下一轮素材建议弱化绝对效果承诺，增加真实使用场景和对比解释。",
            },
            {
                "button": "提取反感点",
                "title": "反感点",
                "desc": "找出用户排斥表达",
                "icon": "N",
                "answer": "当前反感点主要包括：效果表达太绝对、价格解释不足、达人笔记像硬广、使用场景不够真实。建议将素材拆成真实体验、适用人群和长期成本三组验证。",
            },
            {
                "button": "生成素材建议",
                "title": "素材建议",
                "desc": "输出下一轮投放方向",
                "icon": "A",
                "answer": "素材优化建议：优先使用真实用户场景、降低夸张承诺、增加价格构成解释，并把竞品 A 的场景化表达作为参考，但避免直接跟随其价格叙事。",
            },
            {
                "button": "筛选可复用卖点",
                "title": "卖点筛选",
                "desc": "提炼正向评论资产",
                "icon": "G",
                "answer": "可复用卖点包括：包装设计、入门门槛低、售后响应快和长期使用成本可控。建议把这些卖点放进下一轮短视频脚本和达人 brief。",
            },
        ],
    },
    "竞品策略": {
        "subtitle": "负责竞品动态、市场机会、销售话术、对比洞察和策略资产。",
        "metrics": [
            ("竞品提及", "18,420", "+31.2%", "up"),
            ("机会线索", "42", "新增 11", "ok"),
            ("用户痛点", "9", "高频 4", "warn"),
            ("话术包", "3", "待更新", "ok"),
        ],
        "focus": ["提取竞品 A 用户吐槽", "生成门店销售话术", "比较价格敏感人群反馈"],
        "agent_queue": [
            ("竞品 A 新品价格与套餐对比", "证据汇总中", "C-118"),
            ("竞品 B 质量吐槽机会点", "待人工确认", "C-082"),
            ("价格敏感人群反馈提取", "分析中", "S-049"),
            ("门店销售话术包更新", "待生成", "S-061"),
        ],
        "agent_output": "竞品对比卡、销售话术、机会线索、用户痛点清单",
        "risk_lens": "竞品策略视角优先判断负面讨论能否转化为对比机会，重点看竞品优势是否正在重塑用户决策标准。",
        "risk_actions": ["生成竞品对比卡", "更新销售话术", "提取机会线索", "加入竞品周报"],
        "competitor_focus": "重点展示竞品变化带来的销售机会：哪些用户痛点可被本品牌承接，哪些对比点需要补充证据和话术。",
        "report_defaults": {
            "type": "竞品追踪周报",
            "audience": "竞品策略负责人",
            "modules": ["关键结论", "竞品变化", "证据评论", "建议动作"],
        },
        "copilot_commands": [
            {
                "button": "生成竞品话术",
                "title": "销售话术",
                "desc": "生成一线对比话术",
                "icon": "S",
                "answer": "销售话术草稿：如果用户提到竞品 A 套餐更完整，可强调本品牌上手门槛低、售后响应快和长期使用成本更清楚；同时用真实用户评价解释核心功能差异，避免只做价格对比。",
            },
            {
                "button": "提取机会线索",
                "title": "机会线索",
                "desc": "识别可跟进客群",
                "icon": "O",
                "answer": "当前机会线索集中在三类用户：对竞品价格敏感的人、担心质量稳定性的人、希望看到真实横评的人。建议销售团队优先准备价格解释和真实案例材料。",
            },
            {
                "button": "生成对比卡",
                "title": "竞品对比",
                "desc": "生成横向对比卡",
                "icon": "C",
                "answer": "竞品对比卡建议包含：价格与套餐、适用场景、售后响应、真实口碑和长期成本五个维度。竞品 A 胜在场景表达，本品牌可主打上手简单和服务稳定。",
            },
            {
                "button": "写入竞品周报",
                "title": "周报沉淀",
                "desc": "写入竞品追踪周报",
                "icon": "W",
                "answer": "已生成竞品周报段落：竞品 A 新品发布带动价格与套餐讨论，本品牌需补强场景解释和销售话术；竞品 B 促销带来质量吐槽，可作为稳定性对比机会。",
            },
        ],
    },
}


MONITOR_PROJECTS: List[Dict] = [
    {
        "name": "品牌口碑监测",
        "owner": "品牌运营",
        "status": "运行中",
        "coverage": "小红书 / 抖音 / 微博 / 电商评价",
        "signals": 8,
        "description": "监测品牌词、产品词、核心卖点与用户反馈变化。",
    },
    {
        "name": "竞品动态监测",
        "owner": "竞品策略",
        "status": "运行中",
        "coverage": "竞品 A / 竞品 B / 竞品 C",
        "signals": 5,
        "description": "追踪竞品新品、活动、价格讨论和用户吐槽。",
    },
    {
        "name": "春季上新活动复盘",
        "owner": "营销增长",
        "status": "研判中",
        "coverage": "活动话题 / 投放素材 / 达人笔记",
        "signals": 4,
        "description": "评估活动声量、素材接受度、用户反感点和复用卖点。",
    },
    {
        "name": "风险关键词监测",
        "owner": "公关风控",
        "status": "预警中",
        "coverage": "虚假宣传 / 价格虚高 / 质量翻车",
        "signals": 6,
        "description": "识别异常负面提及、高赞负评和潜在危机扩散。",
    },
]


RISK_EVENTS: List[Dict] = [
    {
        "id": "SHU-20260531",
        "level": "高",
        "title": "小红书爆款负面：精华液过敏辣脸",
        "platform": "小红书",
        "trend": "2 小时互动 1,200+，负面评论快速聚集",
        "owner": "公关风控",
        "status": "品牌运营已分发",
        "confidence": 91,
        "summary": "用户集中反馈 NewBrand 光感修护精华液使用后刺痛泛红，事件发生在 618 大促投放窗口，可能触发产品质量风控边界。",
        "drivers": {"产品质量体验": 38, "广告温和承诺": 27, "客服处理不清晰": 19, "大促传播放大": 16},
        "evidence": [
            "用了两次就过敏辣脸，脸颊刺痛泛红，客服还让我继续观察。",
            "广告一直讲温和修护，但我用完明显刺痛，这算不算虚假宣传？",
            "618 还在大推这个精华，建议品牌先把问题解释清楚。",
        ],
        "actions": ["生成回应口径", "整理证据摘要", "同步客服 FAQ", "触发营销熔断"],
    },
    {
        "id": "R-024",
        "level": "高",
        "title": "广告文案引发价格争议",
        "platform": "小红书 / 微博",
        "trend": "负面提及 24 小时增长 168%",
        "owner": "公关风控",
        "status": "待确认升级",
        "confidence": 82,
        "summary": "用户集中质疑新品定价与广告承诺之间存在落差，高赞评论开始向竞品对比扩散。",
        "drivers": {"价格感知落差": 42, "广告承诺过强": 28, "竞品对比": 17, "售后体验": 13},
        "evidence": [
            "价格比竞品贵一截，但宣传里的核心效果没有那么明显。",
            "广告说得太满，实际体验更像常规升级版。",
            "同价位我可能会考虑竞品 A，至少功能解释更清楚。",
        ],
        "actions": ["生成公关回应口径", "分派公关风控", "加入今日风险简报", "追踪 48 小时扩散"],
    },
    {
        "id": "R-021",
        "level": "中",
        "title": "售后响应慢被连续提及",
        "platform": "抖音 / 电商评价",
        "trend": "相关评论连续 3 天上升",
        "owner": "客户体验",
        "status": "处理中",
        "confidence": 74,
        "summary": "用户对售后等待时间和问题一次性解决率不满，暂未形成跨平台大范围扩散。",
        "drivers": {"响应慢": 51, "流程复杂": 23, "退换体验": 16, "客服口径不一致": 10},
        "evidence": [
            "客服回复很慢，等了一天还是让我重新提交材料。",
            "问题不大，但流程太绕，体验被消耗完了。",
            "不同客服说法不一样，不知道该听谁的。",
        ],
        "actions": ["同步客服主管", "生成 FAQ 更新建议", "抽样核查工单", "观察 7 日趋势"],
    },
    {
        "id": "R-018",
        "level": "中",
        "title": "竞品新品带动对比讨论",
        "platform": "小红书 / B站",
        "trend": "竞品 A 提及增长 31%",
        "owner": "竞品策略",
        "status": "待生成话术",
        "confidence": 79,
        "summary": "竞品新品发布后，用户开始比较价格、功能解释和使用场景，本品牌卖点表达需要更明确。",
        "drivers": {"功能对比": 36, "价格对比": 31, "场景解释": 21, "达人测评": 12},
        "evidence": [
            "竞品 A 这次把适用场景讲得更清楚。",
            "两个产品差价不大，但 A 的套餐看起来更完整。",
            "想看一个真实横评，官方图都太像广告了。",
        ],
        "actions": ["生成竞品对比卡", "更新销售话术", "收集达人横评", "加入竞品周报"],
    },
]


COMPETITORS: List[Dict] = [
    {
        "name": "竞品 A",
        "share": 34,
        "sentiment": 68,
        "change": "+31%",
        "signal": "新品发布后声量快速上升，用户关注价格与套装完整度。",
        "opportunity": "强调本品牌使用门槛低、售后响应快和长期成本优势。",
    },
    {
        "name": "竞品 B",
        "share": 22,
        "sentiment": 55,
        "change": "+8%",
        "signal": "促销活动带来短期声量，但质量吐槽同步增加。",
        "opportunity": "在销售话术中对比稳定性与真实用户评价。",
    },
    {
        "name": "竞品 C",
        "share": 14,
        "sentiment": 61,
        "change": "-4%",
        "signal": "近期讨论下降，用户主要关注设计风格与包装。",
        "opportunity": "可吸收其视觉表达优点，但不宜直接跟随价格策略。",
    },
]


REPORTS: List[Dict] = [
    {"name": "突发风控事件处置全周期报告", "type": "风控", "status": "草稿可生成", "owner": "品牌运营", "updated": "刚刚"},
    {"name": "市场情报日报", "type": "日报", "status": "已生成", "owner": "品牌运营", "updated": "今日 09:30"},
    {"name": "广告争议风险简报", "type": "风险", "status": "待审核", "owner": "公关风控", "updated": "今日 10:15"},
    {"name": "竞品 A 新品追踪", "type": "竞品", "status": "生成中", "owner": "竞品策略", "updated": "今日 11:05"},
    {"name": "春季上新活动复盘", "type": "活动", "status": "待补充证据", "owner": "营销增长", "updated": "昨日 18:40"},
]


PAGES = [
    "市场情报总览",
    "Agent 研判中心",
    "风险事件中心",
    "竞品情报雷达",
    "报告中心",
    "新消费评论证据库",
]


ACTIVE_INCIDENT = {
    "id": "SHU-20260531",
    "title": "小红书爆款负面：精华液过敏辣脸",
    "source": "小红书用户 @美妆纠错本",
    "summary": "NewBrand 光感修护精华液被吐槽使用后刺痛泛红、严重辣脸，2 小时内互动量破千，评论区出现多名用户共鸣。",
    "platform": "小红书",
    "product": "光感修护精华液",
    "campaign": "618 大促",
    "trend": "2 小时互动 1,200+，负面评论占比快速上升",
    "risk_level": "高危",
    "confidence": 91,
    "health_drop": "-34",
    "drivers": {"产品质量体验": 38, "广告温和承诺": 27, "客服处理不清晰": 19, "大促传播放大": 16},
    "evidence": [
        "用了两次就过敏辣脸，脸颊刺痛泛红，客服还让我继续观察。",
        "广告一直讲温和修护，但我用完明显刺痛，这算不算虚假宣传？",
        "618 还在大推这个精华，建议品牌先把问题解释清楚。",
    ],
    "agent_basis": [
        "命中产品质量与人身体验相关高危词：过敏、刺痛、辣脸、泛红。",
        "爆款笔记互动速度超过近 7 天同类负面内容 P95 阈值。",
        "事件发生在 618 大促投放窗口，可能影响转化和达人内容可信度。",
        "评论证据显示客服口径不一致，需进入人工审核与统一回应流程。",
    ],
    "response_draft": "建议回应口径：我们已关注到部分用户关于 NewBrand 光感修护精华液使用后刺痛、泛红的反馈。不同肤质对活性成分的耐受度存在差异，我们已启动样本复核和客服专项跟进。建议用户暂停使用并通过官方客服登记肤质、批次和使用情况，我们将在 24 小时内给出处理方案。后续会补充更清晰的敏感肌使用提示和局部测试建议。",
    "marketing_actions": [
        "暂停“敏感肌安心可用”相关 KOC 铺量素材。",
        "将信息流主卖点从“强功效修护”调整为“温和屏障修护 + 先局部测试”。",
        "提取吐槽词：辣脸、刺痛、泛红、客服模板化，更新达人 brief 避免绝对化承诺。",
        "保留真实反馈入口，把高风险评论纳入活动复盘证据。",
    ],
}


def inject_styles() -> None:
    st.markdown(
        """
<style>
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        min-height: 100vh !important;
    }
    :root {
        --bg: #f8fafc;
        --panel: #ffffff;
        --ink: #0f172a;
        --muted: #64748b;
        --soft: #f1f5f9;
        --line: #e2e8f0;
        --line-strong: #cbd5e1;
        --blue: #2563eb;
        --blue-soft: #eff6ff;
        --teal: #14b8a6;
        --teal-soft: #ecfdf5;
        --red: #ef4444;
        --red-soft: #fef2f2;
        --amber: #f59e0b;
        --amber-soft: #fffbeb;
        --green: #10b981;
        --green-soft: #ecfdf5;
        --purple: #8b5cf6;
    }
    .stApp {
        background: var(--bg);
        color: var(--ink);
    }
    header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0;
    }
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
        box-shadow: 8px 0 28px rgba(15, 23, 42, 0.03);
        width: 224px !important;
        min-width: 224px !important;
    }
    section[data-testid="stSidebar"] > div {
        width: 224px !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--ink);
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    section[data-testid="stSidebar"] label {
        color: #475569;
    }
    section[data-testid="stSidebar"] hr {
        border-color: var(--line);
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 8px;
        padding: 0.45rem 0.55rem;
        margin: 0.1rem 0;
        border: 1px solid transparent;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: var(--blue-soft);
        border-left: 3px solid var(--blue);
        color: var(--blue);
    }
    .block-container {
        padding-top: 0.7rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.75rem;
    }
    .app-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.95rem;
        margin-bottom: 1rem;
    }
    .eyebrow {
        color: var(--blue);
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .title {
        font-size: 1.46rem;
        line-height: 1.2;
        font-weight: 760;
        color: var(--ink);
        margin: 0;
    }
    .subtitle {
        color: var(--muted);
        font-size: 0.92rem;
        margin-top: 0.35rem;
    }
    .toolbar {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        justify-content: flex-end;
        flex-wrap: wrap;
    }
    .chip {
        border: 1px solid var(--line);
        background: #fff;
        border-radius: 999px;
        padding: 0.42rem 0.72rem;
        color: var(--muted);
        font-size: 0.78rem;
        white-space: nowrap;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 1rem;
        min-height: 100%;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .panel-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
        gap: 0.5rem;
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 10px;
        background: #fff;
        padding: 0.78rem 0.9rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .panel-title strong {
        font-size: 0.95rem;
        color: var(--ink);
        line-height: 1.35;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    .panel-title span {
        font-size: 0.78rem;
        color: var(--muted);
        line-height: 1.35;
        text-align: right;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    .metric-card {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.95rem 1rem;
        min-height: 118px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        margin-bottom: 0.58rem;
        line-height: 1.35;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    .metric-value {
        color: var(--ink);
        font-weight: 760;
        font-size: 1.5rem;
        line-height: 1.1;
        margin-bottom: 0.7rem;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    .metric-delta {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.2rem 0.45rem;
        font-size: 0.72rem;
        font-weight: 650;
    }
    .delta-up { background: var(--green-soft); color: #047857; }
    .delta-ok { background: var(--blue-soft); color: var(--blue); }
    .delta-warn { background: var(--amber-soft); color: #b45309; }
    .delta-bad { background: var(--red-soft); color: #dc2626; }
    .signal-card {
        border: 1px solid var(--line);
        border-radius: 9px;
        padding: 0.78rem 0.85rem;
        margin-bottom: 0.56rem;
        background: #fff;
        box-shadow: 0 1px 1px rgba(15, 23, 42, 0.02);
    }
    .signal-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.6rem;
        margin-bottom: 0.35rem;
    }
    .signal-title {
        color: var(--ink);
        font-weight: 700;
        font-size: 0.9rem;
        line-height: 1.45;
        overflow-wrap: anywhere;
    }
    .signal-meta {
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.55;
        overflow-wrap: anywhere;
    }
    .tag {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.18rem 0.48rem;
        font-size: 0.72rem;
        font-weight: 700;
        white-space: nowrap;
    }
    .tag-high { background: var(--red-soft); color: #dc2626; }
    .tag-mid { background: var(--amber-soft); color: #b45309; }
    .tag-low { background: var(--blue-soft); color: var(--blue); }
    .tag-run { background: var(--green-soft); color: #047857; }
    .workflow {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.7rem;
        margin: 0.5rem 0 1rem 0;
    }
    .workflow-step {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.75rem;
        min-height: 88px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .workflow-step strong {
        display: block;
        color: var(--ink);
        font-size: 0.84rem;
        margin-bottom: 0.3rem;
    }
    .workflow-step span {
        color: var(--muted);
        font-size: 0.74rem;
        line-height: 1.4;
    }
    .evidence {
        border-left: 3px solid var(--teal);
        padding: 0.55rem 0.65rem;
        margin-bottom: 0.45rem;
        background: #f8fafc;
        color: var(--ink);
        font-size: 0.82rem;
        line-height: 1.55;
        border-radius: 0 8px 8px 0;
    }
    .copilot-box {
        background: #fff;
        border-radius: 10px;
        padding: 1rem;
        color: var(--ink);
        border: 1px solid var(--line);
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .copilot-box strong {
        color: var(--ink);
    }
    .copilot-box p {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
    }
    .copilot-card {
        display: flex;
        gap: 0.7rem;
        align-items: flex-start;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.75rem;
        background: #fff;
        margin-bottom: 0.65rem;
    }
    .copilot-icon {
        width: 2rem;
        height: 2rem;
        border-radius: 9px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--blue-soft);
        color: var(--blue);
        font-weight: 760;
        flex: 0 0 auto;
    }
    .copilot-card strong {
        display: block;
        color: var(--ink);
        font-size: 0.85rem;
        margin-bottom: 0.18rem;
    }
    .copilot-card span {
        display: block;
        color: var(--muted);
        font-size: 0.75rem;
        line-height: 1.4;
    }
    .answer-box {
        border: 1px solid #c7d7fe;
        background: #f7faff;
        border-radius: 10px;
        padding: 0.85rem;
        color: var(--ink);
        font-size: 0.88rem;
        line-height: 1.65;
        margin-top: 0.75rem;
    }
    .incident-card {
        border: 1px solid #fecaca;
        background: linear-gradient(180deg, #fff7f7 0%, #ffffff 100%);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.08);
    }
    .incident-card h3 {
        color: #991b1b;
        font-size: 1rem;
        margin: 0 0 0.45rem;
    }
    .incident-meta-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.6rem;
        margin-top: 0.8rem;
    }
    .incident-meta {
        border: 1px solid #fee2e2;
        background: #fff;
        border-radius: 8px;
        padding: 0.62rem;
    }
    .incident-meta span {
        display: block;
        color: var(--muted);
        font-size: 0.72rem;
    }
    .incident-meta strong {
        color: var(--ink);
        font-size: 0.9rem;
    }
    .task-alert {
        border: 1px solid #fca5a5;
        border-left: 5px solid var(--red);
        background: #fff7f7;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 24px rgba(239, 68, 68, 0.08);
    }
    .task-alert strong {
        color: #991b1b;
    }
    .timeline {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.7rem;
    }
    .timeline-step {
        border: 1px solid var(--line);
        border-radius: 10px;
        background: #fff;
        padding: 0.75rem;
        min-height: 108px;
    }
    .timeline-step strong {
        display: block;
        color: var(--ink);
        font-size: 0.84rem;
        margin-bottom: 0.3rem;
    }
    .timeline-step span {
        color: var(--muted);
        font-size: 0.74rem;
        line-height: 1.45;
    }
    .small-note {
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.5;
    }
    div[data-testid="stRadio"] > label {
        color: var(--muted);
        font-size: 0.8rem;
    }
    div[data-testid="stRadio"] [role="radiogroup"] {
        gap: 0.45rem;
    }
    div[data-testid="stRadio"] input[type="radio"] {
        opacity: 0;
        width: 0;
        height: 0;
        margin: 0;
        position: absolute;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.35rem 0.6rem;
        min-height: 2.15rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
        justify-content: center;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child {
        display: none;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label p {
        color: var(--ink) !important;
        font-weight: 600;
        white-space: normal;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label,
    div[data-testid="stRadio"] [role="radiogroup"] label div,
    div[data-testid="stRadio"] [role="radiogroup"] label span {
        color: var(--ink) !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        background: var(--blue);
        border-color: var(--blue);
    }
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked),
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) *,
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {
        color: #fff !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label {
        justify-content: flex-start;
        box-shadow: none;
        width: 100%;
        min-height: 2.25rem;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        background: var(--blue);
        border-color: var(--blue);
        border-left-color: var(--blue);
        color: #fff !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) * {
        color: #fff !important;
    }
    div[data-testid="stMetric"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.75rem;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: var(--ink) !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
        color: var(--muted) !important;
    }
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextInput"] label *,
    div[data-testid="stTextArea"] label,
    div[data-testid="stTextArea"] label *,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label *,
    div[data-testid="stMultiSelect"] label,
    div[data-testid="stMultiSelect"] label * {
        color: var(--muted) !important;
    }
    .stButton > button {
        border-radius: 8px;
        border: 1px solid var(--line-strong);
        background: #fff;
        color: var(--ink);
        min-height: 2.35rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        white-space: normal;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    .stButton > button:hover {
        border-color: var(--blue);
        color: var(--blue);
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-baseweb="select"] {
        border-radius: 8px;
    }
    .brand-mark {
        display: inline-flex;
        width: 1.75rem;
        height: 1.75rem;
        border-radius: 9px;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #2563eb 0%, #14b8a6 100%);
        color: #fff;
        font-weight: 800;
        margin-right: 0.5rem;
    }
    .sidebar-title {
        display: flex;
        align-items: center;
        font-size: 1.15rem;
        font-weight: 780;
        color: var(--ink);
        margin: 0.25rem 0 0.1rem;
    }
    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        padding: 0.2rem 0 0.75rem;
    }
    .topbar-left {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--muted);
        font-size: 0.82rem;
    }
    .user-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.36rem 0.7rem;
        background: #fff;
        color: var(--ink);
        font-size: 0.78rem;
    }
    .avatar {
        width: 1.45rem;
        height: 1.45rem;
        border-radius: 50%;
        background: #64748b;
        color: #fff;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
    }
    .data-row {
        display: grid;
        grid-template-columns: 1.3fr 0.7fr 0.8fr 0.6fr 0.5fr;
        gap: 0.4rem;
        align-items: center;
        border-bottom: 1px solid var(--line);
        padding: 0.62rem 0;
        font-size: 0.78rem;
    }
    .data-row span {
        min-width: 0;
        overflow-wrap: anywhere;
        line-height: 1.35;
    }
    .data-row.header {
        color: var(--muted);
        font-size: 0.72rem;
        background: #f8fafc;
        border-radius: 8px;
        padding: 0.52rem 0.45rem;
        border-bottom: none;
        margin-bottom: 0.2rem;
    }
    .data-row:not(.header) {
        padding-left: 0.45rem;
        padding-right: 0.45rem;
    }
    .mini-link {
        color: var(--blue);
        font-size: 0.78rem;
        font-weight: 680;
        margin-top: 0.4rem;
    }
    @media (max-width: 980px) {
        .app-header {
            display: block;
        }
        .toolbar {
            justify-content: flex-start;
            margin-top: 0.75rem;
        }
        .workflow {
            grid-template-columns: 1fr;
        }
        .topbar {
            display: block;
        }
    }
</style>
""",
        unsafe_allow_html=True,
    )


def check_api_status() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def query_sample_rag(question: str, top_k: int = 5) -> Optional[Dict]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json={"question": question, "top_k": top_k},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


def get_local_sample_stats() -> Dict:
    path = os.path.join("vector_store", "processed_data.csv")
    if not os.path.exists(path):
        return {"available": False}

    try:
        df = pd.read_csv(path)
        sentiment_counts = df["sentiment"].value_counts().to_dict() if "sentiment" in df else {}
        return {
            "available": True,
            "count": len(df),
            "sentiment_counts": sentiment_counts,
            "columns": list(df.columns),
        }
    except Exception:
        return {"available": False}


def load_evidence_comments(limit: int = 6) -> List[Dict]:
    path = os.path.join("vector_store", "processed_data.csv")
    if not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path)
        if "like_count" in df.columns:
            df = df.sort_values("like_count", ascending=False)
        return df.head(limit).to_dict("records")
    except Exception:
        return []


def ensure_app_state() -> None:
    ensure_work_view()
    if st.session_state.get("nav_page") not in PAGES:
        st.session_state.nav_page = "市场情报总览"
    if "active_incident_id" not in st.session_state:
        st.session_state.active_incident_id = None
    if "trigger_pr_popup" not in st.session_state:
        st.session_state.trigger_pr_popup = False
    if "trigger_marketing_alert" not in st.session_state:
        st.session_state.trigger_marketing_alert = False
    if "pr_response_approved" not in st.session_state:
        st.session_state.pr_response_approved = False
    if "marketing_review_generated" not in st.session_state:
        st.session_state.marketing_review_generated = False
    if "incident_report_ready" not in st.session_state:
        st.session_state.incident_report_ready = False


def dispatch_incident_to_pr() -> None:
    st.session_state.work_view = "公关风控"
    st.session_state.nav_page = "风险事件中心"
    st.session_state.active_incident_id = ACTIVE_INCIDENT["id"]
    st.session_state.trigger_pr_popup = True
    st.session_state.pr_response_approved = False
    st.session_state.trigger_marketing_alert = False
    st.session_state.marketing_review_generated = False
    st.session_state.incident_report_ready = False


def approve_pr_response() -> None:
    st.session_state.pr_response_approved = True
    st.session_state.trigger_marketing_alert = True
    st.session_state.work_view = "营销增长"
    st.session_state.nav_page = "Agent 研判中心"
    st.session_state.active_incident_id = ACTIVE_INCIDENT["id"]
    st.session_state.approved_response_draft = st.session_state.get(
        "pr_response_draft",
        ACTIVE_INCIDENT["response_draft"],
    )


def generate_marketing_review() -> None:
    st.session_state.marketing_review_generated = True
    st.session_state.incident_report_ready = True
    st.session_state.nav_page = "报告中心"


def ensure_work_view() -> str:
    legacy_role_map = {
        "市场运营负责人": "品牌运营",
        "品牌公关": "公关风控",
        "营销广告": "营销增长",
        "销售策略": "竞品策略",
        "管理层": DEFAULT_WORK_VIEW,
    }
    current = st.session_state.get("work_view")
    if current not in ROLE_VIEWS:
        current = legacy_role_map.get(st.session_state.get("role"), DEFAULT_WORK_VIEW)
        st.session_state.work_view = current
    return current


def current_work_view() -> Dict:
    return ROLE_VIEWS[ensure_work_view()]


def list_items(items: List[str]) -> str:
    return "".join(f'<div class="signal-card"><div class="signal-title">{item}</div></div>' for item in items)


def render_header(title: str, subtitle: str, context: str = "新消费品牌演示空间") -> None:
    st.markdown(
        f"""
<div class="app-header">
    <div>
        <div class="eyebrow">InsightOps · 企业市场情报 Agent</div>
        <h1 class="title">{title}</h1>
        <div class="subtitle">{subtitle}</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_topbar() -> None:
    work_view = ensure_work_view()
    view = ROLE_VIEWS[work_view]
    st.markdown(
        f"""
<div class="topbar">
    <div class="topbar-left">
        <span>当前空间：NewBrand 团队</span>
        <span class="chip">当前工作视角：{work_view}</span>
        <span class="chip">近 7 天</span>
        <span class="chip">自定义看板</span>
    </div>
    <div class="toolbar">
        <span class="chip">12 条通知</span>
        <span class="user-pill"><span class="avatar">NB</span> NewBrand 团队</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
    switch_col, hint_col = st.columns([1.65, 1], gap="medium")
    with switch_col:
        st.radio(
            "当前工作视角",
            list(ROLE_VIEWS.keys()),
            horizontal=True,
            key="work_view",
        )
    with hint_col:
        st.markdown(f'<div class="small-note" style="padding-top:1.85rem;">{view["subtitle"]}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str, tone: str) -> None:
    tone_class = {
        "up": "delta-up",
        "ok": "delta-ok",
        "warn": "delta-warn",
        "bad": "delta-bad",
    }.get(tone, "delta-ok")
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div>
    <span class="metric-delta {tone_class}">{delta}</span>
</div>
""",
        unsafe_allow_html=True,
    )


def panel_title(title: str, caption: str = "") -> None:
    st.markdown(
        f"""
<div class="panel-title">
    <strong>{title}</strong>
    <span>{caption}</span>
</div>
""",
        unsafe_allow_html=True,
    )


def render_monitor_table() -> None:
    rows = [
        ("新品上市舆情监测", "运行中", "326", "2", "↗"),
        ("竞品动态追踪", "运行中", "512", "1", "↗"),
        ("价格与促销监测", "运行中", "278", "0", "↗"),
        ("社媒声量监测", "运行中", "689", "3", "↗"),
        ("行业政策跟踪", "已暂停", "120", "1", "↗"),
    ]
    html = [
        '<div class="data-row header"><span>项目名称</span><span>状态</span><span>今日新增</span><span>风险</span><span>操作</span></div>'
    ]
    for name, status, added, risk, action in rows:
        tag_class = "tag-run" if status == "运行中" else "tag-low"
        risk_color = "#dc2626" if risk != "0" else "#475569"
        html.append(
            f"""
<div class="data-row">
    <span>{name}</span>
    <span><span class="tag {tag_class}">{status}</span></span>
    <span>{added}</span>
    <span style="color:{risk_color}; font-weight:700;">{risk}</span>
    <span style="color:var(--blue); font-weight:700;">{action}</span>
</div>
"""
        )
    html.append('<div class="mini-link">查看全部项目 →</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


def render_competitor_mini_radar() -> None:
    categories = ["产品力", "价格力", "渠道力", "营销力", "口碑力", "创新力"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=[72, 68, 61, 76, 70, 58],
            theta=categories,
            fill="toself",
            name="本品牌",
            line_color="#14b8a6",
            fillcolor="rgba(20, 184, 166, 0.12)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[82, 74, 67, 71, 62, 78],
            theta=categories,
            fill="toself",
            name="竞品 A",
            line_color="#2563eb",
            fillcolor="rgba(37, 99, 235, 0.10)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[65, 82, 72, 58, 64, 70],
            theta=categories,
            fill="toself",
            name="竞品 B",
            line_color="#8b5cf6",
            fillcolor="rgba(139, 92, 246, 0.10)",
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=12, r=12, t=12, b=12),
        polar=dict(radialaxis=dict(visible=False, range=[0, 100]), bgcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.12, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#475569", size=11),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_trend_chart() -> None:
    trend_df = pd.DataFrame(
        {
            "日期": ["05-08", "05-09", "05-10", "05-11", "05-12", "05-13", "05-14"],
            "本品牌": [3500, 5600, 3900, 5200, 6800, 5000, 6700],
            "竞品 A": [2100, 3000, 1900, 2700, 3400, 2500, 3000],
            "竞品 B": [900, 1300, 850, 1100, 1250, 980, 1180],
        }
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df["日期"], y=trend_df["本品牌"], name="本品牌", mode="lines", line=dict(color="#14b8a6", width=3), fill="tozeroy", fillcolor="rgba(20, 184, 166, .08)"))
    fig.add_trace(go.Scatter(x=trend_df["日期"], y=trend_df["竞品 A"], name="竞品 A", mode="lines", line=dict(color="#2563eb", width=2)))
    fig.add_trace(go.Scatter(x=trend_df["日期"], y=trend_df["竞品 B"], name="竞品 B", mode="lines", line=dict(color="#8b5cf6", width=2)))
    fig.update_layout(
        height=270,
        margin=dict(l=8, r=8, t=8, b=8),
        legend=dict(orientation="h", y=1.08, x=0.58),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#e2e8f0"),
        font=dict(color="#475569", size=11),
    )
    st.plotly_chart(fig, use_container_width=True)


def risk_tag(level: str) -> str:
    if level == "高":
        return '<span class="tag tag-high">高风险</span>'
    if level == "中":
        return '<span class="tag tag-mid">中风险</span>'
    return '<span class="tag tag-low">低风险</span>'


def render_signal_card(event: Dict) -> None:
    st.markdown(
        f"""
<div class="signal-card">
    <div class="signal-top">
        <div class="signal-title">{event["title"]}</div>
        {risk_tag(event["level"])}
    </div>
    <div class="signal-meta">
        {event["trend"]} · {event["platform"]}<br>
        负责人：{event["owner"]} · 状态：{event["status"]} · AI 置信度：{event["confidence"]}%
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_incident_dispatch_card() -> None:
    incident = ACTIVE_INCIDENT
    st.markdown(
        f"""
<div class="incident-card">
    <div class="signal-top">
        <div>
            <div class="eyebrow">Risk-LLM · 实时高危信号拦截</div>
            <h3>{incident["title"]}</h3>
        </div>
        <span class="tag tag-high">{incident["risk_level"]}</span>
    </div>
    <div class="signal-meta">
        {incident["source"]}：{incident["summary"]}
    </div>
    <div class="incident-meta-grid">
        <div class="incident-meta"><span>传播速度</span><strong>{incident["trend"]}</strong></div>
        <div class="incident-meta"><span>风险定级</span><strong>{incident["risk_level"]} · {incident["confidence"]}%</strong></div>
        <div class="incident-meta"><span>业务窗口</span><strong>{incident["campaign"]}</strong></div>
        <div class="incident-meta"><span>舆情健康度</span><strong>{incident["health_drop"]} pts</strong></div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.warning(
        "Agent 研判依据：命中产品质量与人身体验高危词，互动速度超过同类负面 P95 阈值，"
        "且发生在 618 大促投放窗口，建议立即进入公关风控工单。"
    )
    st.button(
        "一键介入并分发至公关",
        key="dispatch_active_incident",
        type="primary",
        use_container_width=True,
        on_click=dispatch_incident_to_pr,
    )


def render_pr_task_panel() -> None:
    incident = ACTIVE_INCIDENT
    st.markdown(
        f"""
<div class="task-alert">
    <strong>新任务紧急提醒：舆情危机介入</strong><br>
    事件来源：品牌运营团队一键分发 · 工单编号：{incident["id"]}<br>
    危机摘要：{incident["summary"]}
</div>
""",
        unsafe_allow_html=True,
    )
    st.error(f"高危事件：{incident['source']} 发布关于“{incident['product']}”过敏辣脸投诉，已进入一级响应窗口。")
    st.warning(
        "Agent 研判结论：已触发产品质量风控边界，舆情健康度急剧下跌，"
        "预估影响本周大促转化，建议立即启动一级回应口径并同步营销侧熔断策略。"
    )
    if st.button("立即进入工单处理", key="open_pr_ticket", type="primary"):
        st.session_state.trigger_pr_popup = False
        st.rerun()


def render_pr_response_workspace() -> None:
    incident = ACTIVE_INCIDENT
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_title("人机协同回应口径", "HITL 审核")
    left, right = st.columns([1.15, 0.85], gap="medium")
    with left:
        draft = st.text_area(
            "Agent 生成草稿",
            value=incident["response_draft"],
            height=180,
            key="pr_response_draft",
        )
        st.button(
            "审核通过并下发给营销增长",
            key="approve_pr_response",
            type="primary",
            use_container_width=True,
            on_click=approve_pr_response,
        )
    with right:
        st.markdown("**口径策略**")
        for strategy in ["先承认已关注反馈", "避免直接否认用户体验", "引导登记肤质与批次", "承诺 24 小时处理窗口"]:
            st.markdown(f'<div class="evidence">{strategy}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_marketing_alert_panel() -> None:
    incident = ACTIVE_INCIDENT
    st.markdown('<div class="task-alert">', unsafe_allow_html=True)
    st.markdown(
        f"<strong>大促活动熔断与素材优化警报</strong><br>{incident['product']} 负面事件已由公关风控下发，营销侧需调整 618 大促投放策略。",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.info("Agent 建议：暂停高风险素材，弱化绝对化功效承诺，改用“温和修护 + 局部测试 + 客服登记”表达。")

    cols = st.columns(2)
    for idx, action in enumerate(incident["marketing_actions"]):
        with cols[idx % 2]:
            st.checkbox(action, value=True, key=f"marketing_action_{idx}")

    st.button(
        "生成活动复盘骨架并进入报告中心",
        key="generate_marketing_review",
        type="primary",
        on_click=generate_marketing_review,
    )


def render_incident_report_preview() -> None:
    incident = ACTIVE_INCIDENT
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_title("突发风控事件处置全周期报告", "管理层可读")
    st.markdown(f"#### {incident['title']}")
    st.markdown(f'<div class="small-note">{incident["summary"]}</div>', unsafe_allow_html=True)
    timeline = [
        ("监测拦截", "Risk-LLM 捕捉小红书爆款负面，判定为高危产品体验事件。"),
        ("Agent 研判", "提取过敏、刺痛、辣脸、客服模板化等证据，评估大促影响。"),
        ("公关处置", "生成一级回应口径，由公关风控审核后下发客服和社媒团队。"),
        ("营销调整", "暂停高风险素材，调整达人 brief 和信息流卖点表达。"),
        ("资产沉淀", "形成风控事件报告、复盘骨架和后续观察指标。"),
    ]
    html = ['<div class="timeline">']
    for title, desc in timeline:
        html.append(f'<div class="timeline-step"><strong>{title}</strong><span>{desc}</span></div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_workflow() -> None:
    steps = [
        ("配置监测", "品牌词、竞品词、活动词、风险词"),
        ("信号发现", "识别异常声量、负面扩散与新兴议题"),
        ("Agent 研判", "归因、证据链、置信度和影响面"),
        ("业务处置", "分派负责人、生成口径和行动建议"),
        ("报告沉淀", "日报、周报、复盘与策略资产"),
    ]
    html = ['<div class="workflow">']
    for title, desc in steps:
        html.append(f'<div class="workflow-step"><strong>{title}</strong><span>{desc}</span></div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_market_dashboard() -> None:
    work_view = ensure_work_view()
    view = current_work_view()
    render_header("市场情报总览", f"当前工作视角：{work_view}。{view['subtitle']}")

    metric_cols = st.columns(5)
    metrics = view["metrics"] + [("情报覆盖平台", "42", "新增 3", "ok")]
    for col, metric in zip(metric_cols, metrics):
        with col:
            metric_card(*metric)

    st.write("")
    main_col, copilot_col = st.columns([2.65, 1], gap="medium")

    with main_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("实时高危信号拦截队列", "主动拦截 -> 自动研判 -> 一键分发")
        render_incident_dispatch_card()
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

        row1_left, row1_mid, row1_right = st.columns([1.18, 1.08, 0.98], gap="medium")
        with row1_left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            panel_title("监测项目", "全部项目")
            render_monitor_table()
            st.markdown("</div>", unsafe_allow_html=True)

        with row1_mid:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            panel_title("风险信号", "全部 7")
            for event in RISK_EVENTS:
                render_signal_card(event)
            st.markdown('<div class="mini-link">查看全部风险 →</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with row1_right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            panel_title("竞品雷达", "能力对比")
            render_competitor_mini_radar()
            st.markdown('<div class="mini-link">查看竞品对比 →</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        row2_left, row2_right = st.columns([1.35, 1], gap="medium")
        with row2_left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            panel_title("情报趋势", "声量趋势")
            render_trend_chart()
            st.markdown("</div>", unsafe_allow_html=True)

        with row2_right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            panel_title("证据速览", "最新证据")
            evidence_rows = [
                ("抖音", "用户讨论新品包装设计", "正面", "2 分钟前", "tag-run"),
                ("小红书", "KOL 测评中提到竞品降价", "中性", "15 分钟前", "tag-mid"),
                ("微博", "用户反馈产品口感问题", "负面", "32 分钟前", "tag-high"),
            ]
            for platform, text, tone, time, tag_class in evidence_rows:
                st.markdown(
                    f"""
<div class="signal-card">
    <div class="signal-top">
        <div class="signal-title">{platform} · {text}</div>
        <span class="tag {tag_class}">{tone}</span>
    </div>
    <div class="signal-meta">{time}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
            st.markdown('<div class="mini-link">查看全部证据 →</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("监测到处置闭环", "企业业务流程")
        render_workflow()
        st.markdown("</div>", unsafe_allow_html=True)

    with copilot_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("今日待处理事项", work_view)
        st.markdown(list_items(view["focus"]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        render_copilot("首页总览", RISK_EVENTS[0], compact=True)


def render_agent_center() -> None:
    work_view = ensure_work_view()
    view = current_work_view()
    render_header(
        "Agent 研判中心",
        f"当前工作视角：{work_view}。Agent 会按该视角重排信号队列、待办任务和建议输出物。",
    )

    if work_view == "营销增长" and st.session_state.get("trigger_marketing_alert"):
        render_marketing_alert_panel()
        st.write("")

    stage_cols = st.columns(4)
    stage_metrics = [
        ("捕捉信号", "23", "较昨日 +6", "up"),
        ("待确认研判", "8", "需人工复核", "warn"),
        ("已生成建议", "12", work_view, "ok"),
        ("推荐输出物", "6", "按视角预设", "ok"),
    ]
    for col, metric in zip(stage_cols, stage_metrics):
        with col:
            metric_card(*metric)

    st.write("")
    left, right = st.columns([1.15, 1.85])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("信号队列", f"{work_view}优先级")
        for title, status, code in view["agent_queue"]:
            st.markdown(
                f"""
<div class="signal-card">
    <div class="signal-title">{title}</div>
    <div class="signal-meta">编号：{code} · 状态：{status}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        st.markdown(f'<div class="answer-box">建议输出物：{view["agent_output"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        selected = RISK_EVENTS[0]
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("当前重点研判", selected["id"])
        st.markdown(f"### {selected['title']}")
        st.markdown(f'<div class="small-note">{selected["summary"]}</div>', unsafe_allow_html=True)

        chart_cols = st.columns([1.1, 1])
        with chart_cols[0]:
            driver_df = pd.DataFrame(
                {"归因": list(selected["drivers"].keys()), "占比": list(selected["drivers"].values())}
            )
            fig = px.bar(
                driver_df,
                x="占比",
                y="归因",
                orientation="h",
                color="占比",
                color_continuous_scale=["#c7d7fe", "#246bfe"],
            )
            fig.update_layout(height=280, margin=dict(l=8, r=8, t=12, b=8), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with chart_cols[1]:
            st.markdown("**证据链**")
            for evidence in selected["evidence"]:
                st.markdown(f'<div class="evidence">{evidence}</div>', unsafe_allow_html=True)

        st.markdown("**建议动作**")
        action_cols = st.columns(4)
        for col, action in zip(action_cols, view["risk_actions"]):
            with col:
                st.button(action, key=f"agent_{work_view}_{action}", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    render_copilot(f"高风险事件 {RISK_EVENTS[0]['id']}", RISK_EVENTS[0])


def render_risk_center() -> None:
    work_view = ensure_work_view()
    view = current_work_view()
    render_header("风险事件中心", f"当前工作视角：{work_view}。同一风险事件会按不同业务职责给出解释重点和处置动作。")

    if st.session_state.get("trigger_pr_popup"):
        render_pr_task_panel()

    list_col, detail_col = st.columns([1, 1.6])

    with list_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("风险列表", "按紧急度排序")
        labels = [f'{event["id"]} · {event["title"]}' for event in RISK_EVENTS]
        active_id = st.session_state.get("active_incident_id")
        default_index = next((idx for idx, event in enumerate(RISK_EVENTS) if event["id"] == active_id), 0)
        selected_label = st.radio(
            "选择风险事件",
            labels,
            index=default_index,
            key=f"risk_event_selector_{active_id or 'default'}",
            label_visibility="collapsed",
        )
        selected_index = labels.index(selected_label)
        for event in RISK_EVENTS:
            render_signal_card(event)
        st.markdown("</div>", unsafe_allow_html=True)

    event = RISK_EVENTS[selected_index]
    with detail_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("事件详情", event["id"])
        top = st.columns(4)
        top[0].metric("风险等级", event["level"])
        top[1].metric("AI 置信度", f'{event["confidence"]}%')
        top[2].metric("负责人", event["owner"])
        top[3].metric("状态", event["status"])
        st.markdown(f"#### {event['title']}")
        st.markdown(f'<div class="small-note">{event["summary"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box"><strong>{work_view}解释重点：</strong>{view["risk_lens"]}</div>', unsafe_allow_html=True)

        st.write("")
        driver_df = pd.DataFrame(
            {"原因": list(event["drivers"].keys()), "占比": list(event["drivers"].values())}
        )
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=driver_df["原因"],
                    values=driver_df["占比"],
                    hole=0.58,
                    marker=dict(colors=["#246bfe", "#12a594", "#f59e0b", "#ef4444"]),
                )
            ]
        )
        fig.update_layout(height=260, margin=dict(l=8, r=8, t=8, b=8), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**评论证据**")
        for evidence in event["evidence"]:
            st.markdown(f'<div class="evidence">{evidence}</div>', unsafe_allow_html=True)

        st.markdown("**处置动作**")
        cols = st.columns(4)
        for col, action in zip(cols, view["risk_actions"]):
            with col:
                st.button(action, key=f"{event['id']}_{work_view}_{action}", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if event["id"] == ACTIVE_INCIDENT["id"]:
        st.write("")
        render_pr_response_workspace()
        if st.session_state.get("pr_response_approved"):
            st.success("回应口径已审核通过，并已下发给营销增长进入活动熔断与素材优化流程。")

    st.write("")
    render_copilot(f"风险事件 {event['id']}", event)


def render_competitor_radar() -> None:
    work_view = ensure_work_view()
    view = current_work_view()
    render_header("竞品情报雷达", f"当前工作视角：{work_view}。竞品动态会被转译成该视角最需要的机会、风险或行动资产。")

    comp_df = pd.DataFrame(COMPETITORS)
    top_cols = st.columns(3)
    for col, comp in zip(top_cols, COMPETITORS):
        with col:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">{comp["name"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">声量 {comp["share"]}%</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="metric-delta delta-up">{comp["change"]}</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="small-note" style="margin-top:.7rem;">{comp["signal"]}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("声量与口碑对比", "Mock 数据")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="声量份额", x=comp_df["name"], y=comp_df["share"], marker_color="#246bfe"))
        fig.add_trace(go.Scatter(name="正向口碑", x=comp_df["name"], y=comp_df["sentiment"], marker_color="#12a594"))
        fig.update_layout(height=360, margin=dict(l=8, r=8, t=24, b=8), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("可转化机会", work_view)
        st.markdown(f'<div class="answer-box">{view["competitor_focus"]}</div>', unsafe_allow_html=True)
        for comp in COMPETITORS:
            if work_view == "营销增长":
                opportunity = f"素材启发：{comp['signal']} 可拆解其达人表达、促销包装和用户反感点，作为下一轮投放测试假设。"
            elif work_view == "公关风控":
                opportunity = f"风险解释：{comp['signal']} 需观察竞品对比是否放大本品牌负面，优先准备事实边界和客服口径。"
            elif work_view == "品牌运营":
                opportunity = f"协同事项：{comp['signal']} 建议同步至日报，并判断是否分派给营销增长或竞品策略继续跟进。"
            else:
                opportunity = comp["opportunity"]
            st.markdown(
                f"""
<div class="signal-card">
    <div class="signal-title">{comp["name"]}</div>
    <div class="signal-meta">{opportunity}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    render_copilot("竞品雷达", RISK_EVENTS[2])


def render_report_center() -> None:
    work_view = ensure_work_view()
    view = current_work_view()
    defaults = view["report_defaults"]
    render_header("报告中心", f"当前工作视角：{work_view}。报告生成器会自动预设报告类型、目标读者和默认模块。")

    if st.session_state.get("active_incident_id") == ACTIVE_INCIDENT["id"]:
        render_incident_report_preview()
        st.write("")

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("报告列表", "按更新时间")
        report_df = pd.DataFrame(REPORTS)
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("报告生成器", f"{work_view}预设")
        st.markdown('<div class="small-note">管理层保留为报告受众和老板版简报，不作为日常可切换工作视角。</div>', unsafe_allow_html=True)
        report_types = ["市场情报日报", "风险事件简报", "竞品追踪周报", "营销活动复盘", "管理层简报"]
        audiences = ["品牌运营负责人", "公关风控负责人", "营销增长负责人", "竞品策略负责人", "管理层"]
        module_options = ["关键结论", "风险事件", "竞品变化", "证据评论", "建议动作", "下周观察点"]
        report_type = st.selectbox(
            "报告类型",
            report_types,
            index=report_types.index(defaults["type"]),
            key=f"report_type_{work_view}",
        )
        audience = st.selectbox(
            "目标读者",
            audiences,
            index=audiences.index(defaults["audience"]),
            key=f"audience_{work_view}",
        )
        include = st.multiselect(
            "包含模块",
            module_options,
            default=[module for module in defaults["modules"] if module in module_options],
            key=f"modules_{work_view}",
        )
        st.markdown(f'<div class="small-note">默认输出物：{view["agent_output"]}</div>', unsafe_allow_html=True)
        draft_key = f"report_draft_{work_view}"
        if st.button("生成报告草稿", use_container_width=True, type="primary"):
            st.session_state[draft_key] = (
                f"{report_type}草稿已生成，面向{audience}，包含{len(include)}个模块。"
                f"当前按“{work_view}”视角组织内容，建议重点呈现广告争议风险、竞品 A 新品影响和售后体验观察点。"
            )
        if st.session_state.get(draft_key):
            st.markdown(f'<div class="answer-box">{st.session_state[draft_key]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    render_copilot("报告中心", RISK_EVENTS[0])


def render_sample_data_lab() -> None:
    work_view = ensure_work_view()
    render_header("新消费评论证据库", f"当前工作视角：{work_view}。本页展示护肤/美妆品牌评论证据与本地 RAG 技术链路。")

    stats = get_local_sample_stats()
    api_online = check_api_status()

    cols = st.columns(4)
    cols[0].metric("后端 API", "在线" if api_online else "离线")
    cols[1].metric("证据评论", stats.get("count", 0) if stats.get("available") else "未加载")
    cols[2].metric("数据定位", "护肤/美妆评论")
    cols[3].metric("向量库", "TF-IDF 本地检索")

    st.markdown(
        """
<div class="small-note">
说明：当前证据库已替换为新消费护肤/美妆品牌评论 Mock 数据，用于支撑过敏辣脸、广告承诺、客服口径、
大促投放等市场情报场景。它用于演示评论检索、情感分析、主题聚类和 RAG 查询链路。
</div>
""",
        unsafe_allow_html=True,
    )

    if stats.get("available") and stats.get("sentiment_counts"):
        st.write("")
        sentiment_df = pd.DataFrame(
            {"情感": list(stats["sentiment_counts"].keys()), "数量": list(stats["sentiment_counts"].values())}
        )
        fig = px.bar(
            sentiment_df,
            x="情感",
            y="数量",
            color="情感",
            color_discrete_map={"neutral": "#94a3b8", "positive": "#12a594", "negative": "#ef4444"},
        )
        fig.update_layout(
            height=320,
            margin=dict(l=8, r=8, t=16, b=8),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#475569", size=11),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#e2e8f0"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("")
    evidence = load_evidence_comments(limit=6)
    if evidence:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title("高互动评论证据", "来自本地新消费证据库")
        for row in evidence:
            sentiment = row.get("sentiment", "neutral")
            tag_class = {"positive": "tag-run", "negative": "tag-high", "neutral": "tag-mid"}.get(sentiment, "tag-low")
            platform = row.get("platform", "社媒")
            product = row.get("product", "产品")
            likes = int(row.get("like_count", 0))
            st.markdown(
                f"""
<div class="signal-card">
    <div class="signal-top">
        <div class="signal-title">{platform} · {product}</div>
        <span class="tag {tag_class}">{sentiment}</span>
    </div>
    <div class="signal-meta">{row.get("content", "")}<br>互动：{likes} 赞 · 风险标签：{row.get("risk_label", "观察")}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_title("新消费 RAG 查询", "仅在后端服务启动时可用")
    question = st.text_input("查询新消费评论证据库", value="精华液过敏辣脸相关负面情绪主要集中在哪些方面？")
    if st.button("查询证据库", type="primary"):
        result = query_sample_rag(question)
        if result:
            st.markdown(f'<div class="answer-box">{result.get("answer", "")}</div>', unsafe_allow_html=True)
            st.caption(f"检索到 {result.get('retrieval_count', 0)} 条新消费评论。")
        else:
            st.warning("后端未启动或查询失败。可先运行 start.bat 启动 API 服务。")
    st.markdown("</div>", unsafe_allow_html=True)


def render_copilot(context: str, event: Dict, compact: bool = False) -> None:
    work_view = ensure_work_view()
    view = current_work_view()
    answer_key = f"copilot_answer_{context}_{work_view}"
    st.markdown('<div class="copilot-box">', unsafe_allow_html=True)
    st.markdown("<strong>✦ Copilot</strong>", unsafe_allow_html=True)
    st.markdown(
        f"<p>您好，NewBrand 团队。当前上下文：{context}。当前工作视角：{work_view}。我会优先生成该视角最需要的动作、报告或话术资产。</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    for command in view["copilot_commands"]:
        st.markdown(
            f"""
<div class="copilot-card">
    <span class="copilot-icon">{command["icon"]}</span>
    <div><strong>{command["title"]}</strong><span>{command["desc"]}</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button(command["button"], key=f"{context}_{work_view}_{command['button']}", use_container_width=True):
            st.session_state[answer_key] = command["answer"]

    tags = st.columns(4)
    for tag_col, tag in zip(tags, ["全网", "社媒", "新闻", "行业报告"]):
        with tag_col:
            st.button(tag, key=f"{context}_{work_view}_tag_{tag}", use_container_width=True)

    custom = st.text_input(
        "输入业务指令",
        placeholder="输入你的问题或指令，Shift + Enter 换行",
        key=f"copilot_input_{context}_{work_view}",
    )
    if st.button("发送给 Copilot", key=f"send_{context}_{work_view}", type="primary"):
        if custom.strip():
            st.session_state[answer_key] = (
                f"已基于“{event['title']}”和“{work_view}”视角生成建议：{view['risk_lens']}"
                f"下一步建议围绕“{custom.strip()}”输出可交付材料，并沉淀为{view['agent_output']}。"
            )
        else:
            st.session_state[answer_key] = "请输入一个具体业务指令，例如生成回应口径、整理证据摘要、生成竞品话术或加入今日简报。"

    if st.session_state.get(answer_key):
        st.markdown(f'<div class="answer-box">{st.session_state[answer_key]}</div>', unsafe_allow_html=True)


def render_sidebar() -> str:
    st.sidebar.markdown(
        """
<div class="sidebar-title"><span class="brand-mark">IO</span> InsightOps</div>
<div class="small-note">企业市场情报 Agent</div>
""",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "导航",
        PAGES,
        key="nav_page",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("数据更新")
    st.sidebar.markdown(
        """
- 新消费品牌 Mock 数据
- 新消费评论证据库
- 2 分钟前同步
"""
    )
    st.sidebar.markdown("---")
    api_online = check_api_status()
    if api_online:
        st.sidebar.success("API 在线")
    else:
        st.sidebar.warning("API 未启动")
    return page


def main() -> None:
    inject_styles()
    ensure_app_state()
    page = render_sidebar()
    render_topbar()

    if page == "市场情报总览":
        render_market_dashboard()
    elif page == "Agent 研判中心":
        render_agent_center()
    elif page == "风险事件中心":
        render_risk_center()
    elif page == "竞品情报雷达":
        render_competitor_radar()
    elif page == "报告中心":
        render_report_center()
    else:
        render_sample_data_lab()


if __name__ == "__main__":
    main()
