# -*- coding: utf-8 -*-
"""
结构化查询工具 —— 查询类工具 #2（Day 7 从游戏版改造为客服版）

职责：精确查询结构化数据（订单表），不走向量检索。

和 search_kb 的分工：
    - search_kb：语义检索，回答"开放性问题"（退款多久到账、发票怎么开）
    - query_info：精确查询，回答"事实性问题"（订单到哪了？金额多少？）

为什么需要它：
    - 知识库是"规则与政策"，订单是"用户专属数据"，两套数据源必须分开
    - 真实电商客服对接订单系统（OMS），一次精确查询 > 十次语义检索
    - 找不到订单时明确返回 NOT_FOUND，让模型如实说"查不到"，不编造

数据说明：data/orders.json 是演示用模拟订单（真实系统里这是订单服务 API 的返回）。
"""
from __future__ import annotations

import json
import os
from typing import Optional

from tools.registry import register_tool

ORDERS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "orders.json")

# 中文友好字段 → 订单 JSON 字段
_FIELD_ALIAS = {
    "状态": "status",
    "物流": "logistics",
    "金额": "amount",
    "价格": "amount",
    "商品": "product",
    "产品": "product",
    "下单时间": "create_time",
    "时间": "create_time",
}


def _load_orders() -> list[dict]:
    if not os.path.exists(ORDERS_PATH):
        return []
    with open(ORDERS_PATH, encoding="utf-8") as f:
        return json.load(f)["orders"]


@register_tool
def query_info(entity: str, field: Optional[str] = None) -> str:
    """精确查询订单信息（状态、物流、金额等）。

    当用户提供订单号（12 位以上数字）询问订单状态、物流进度、订单金额时使用；
    订单号不存在时返回 NOT_FOUND，请如实告知用户并建议核实订单号。
    """
    entity = entity.strip()
    orders = _load_orders()

    # 支持带前缀的写法："订单 JD20260815001" / "JD20260815001"
    order_id = entity
    for prefix in ("订单", "订单号", "单号"):
        if entity.startswith(prefix):
            order_id = entity[len(prefix):].strip()
            break

    order = next((o for o in orders if o["order_id"] == order_id), None)
    if order is None:
        return f"[NOT_FOUND] 未找到订单 {order_id}。请用户核对订单号，或通过「我的-订单」自助查询。"

    if field:
        fname = _FIELD_ALIAS.get(field.strip(), field.strip())
        if fname in order:
            return f"[查询结果] 订单 {order_id}（{order['product']}）的{fname}是：{order[fname]}"
        return (
            f"[查询结果] 订单 {order_id}（{order['product']}）没有字段「{field}」，"
            f"可查字段：状态、物流、金额、商品、下单时间"
        )

    # 无字段 → 返回完整摘要
    return (
        f"[查询结果] 订单 {order['order_id']}：{order['product']}，"
        f"金额 ¥{order['amount']}，状态：{order['status']}，{order['logistics']}"
    )
