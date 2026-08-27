# -*- coding: utf-8 -*-
"""
共享的数据分类工具 —— 数据集管线（管线A 候选提取 / 管线B 会话采样）复用。

职责：
    1. classify(text)   按主题关键词把文本归入 9 大客服主题
    2. QUESTION_PAT     问句信号正则（判断"用户是否在提问"）
    3. PLACEHOLDER_PAT  脱敏占位符检测（[数字x]/[姓名x]/[站点x]...）
    4. is_orders_text   是否包含订单号（JD...）——用于工具路由推断

为什么抽成共享模块：
    - 数据集两套管线（FAQ 提取、多轮采样）此前各自维护一份主题表和解析逻辑，
      改一个忘了另一个 → 分类口径漂移。现在单一事实来源，改一处全生效。
    - 主题分类口径统一后，faq_candidates 的 topic 与 kb.json 的 category
      才能稳定对齐（variants 抽取依赖这个对齐）。
"""
from __future__ import annotations

import re

# 问句信号（与 build_kb_from_jddc.py 原口径一致）
QUESTION_PAT = re.compile(r"[?？]|吗|怎么|多少|多久|什么时候|能否|可以.*吗|为什么|咋|如何|行不行|可不可以")

# JDDC 脱敏占位符：[数字x] [姓名x] [站点x] [日期x] [金额x] [订单编号:...]
PLACEHOLDER_PAT = re.compile(r"\[[^\[\]]{1,16}\]")

# 订单号：JD + 数字（演示订单数据用 JD + 11 位数字）
ORDER_PAT = re.compile(r"JD\d{6,}")

# 主题分类（关键词 → 主题名）。注意：与 kb.json 的 category 命名一致，
# 这样 faq_candidates.topic 才能直接对齐到 kb 条目的 category。
TOPICS: list[tuple[str, list[str]]] = [
    ("物流发货", ["发货", "物流", "快递", "配送", "到货", "什么时候到", "多久到", "未收到", "没收到", "没到", "揽收"]),
    ("退货退款", ["退货", "退款", "退钱", "退掉", "申请退", "不想要", "不喜欢", "退换"]),
    ("发票", ["发票", "开票", "报销"]),
    ("换货维修", ["换货", "更换", "维修", "坏了", "质量问题", "破损", "故障", "保修", "质保"]),
    ("保价差价", ["保价", "差价", "补差价", "降价"]),
    ("订单修改", ["改地址", "修改地址", "取消订单", "取消", "改单", "订单号"]),
    ("优惠活动", ["优惠券", "优惠", "活动", "满减", "促销", "赠品"]),
    ("售后政策", ["售后", "七天", "无理由"]),
    ("人工客服", ["人工", "转人工", "投诉", "客服电话", "电话", "客服"]),
]


def classify(text: str) -> str:
    """按主题关键词归类，返回主题名或 '其他'。"""
    for topic, kws in TOPICS:
        if any(kw in text for kw in kws):
            return topic
    return "其他"


def has_placeholder(text: str) -> bool:
    """文本里是否残留脱敏占位符（如 [数字x]）。"""
    return bool(PLACEHOLDER_PAT.search(text))
