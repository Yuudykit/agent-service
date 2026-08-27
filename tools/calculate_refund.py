# -*- coding: utf-8 -*-
"""
退款计算工具 —— 计算类工具 #1（Day 7 新增）

职责：根据订单信息计算退款金额，输出可执行的结论。

和 query_info 的分工：
    - query_info：查"订单现在什么状态"（读数据）
    - calculate_refund：算"这单能退多少钱"（算业务）

计算规则（演示版，真实系统由售后规则引擎计算）：
    应退金额 = 实付金额 = 订单金额 - 已用优惠券
    特殊场景：
      - 质量问题退全款（含运费）：应退 = 订单金额（运费险/质量原因由平台承担）
      - 部分商品退货：按退货商品金额比例计算（演示简化，仅整单退款）

输出同时给出"退款去向"说明（原路返回），与知识库政策保持一致。
"""
from __future__ import annotations

from typing import Optional

from tools.query_info import _load_orders
from tools.registry import register_tool


@register_tool
def calculate_refund(order_id: str, quality_issue: bool = False) -> str:
    """计算订单可退金额。

    当用户询问"能退多少钱/退款金额是多少"且提供了订单号时使用；
    quality_issue=True 表示商品质量问题（退全款含运费）。
    订单不存在返回 NOT_FOUND。
    """
    order_id = order_id.strip()
    order = next((o for o in _load_orders() if o["order_id"] == order_id), None)
    if order is None:
        return f"[NOT_FOUND] 未找到订单 {order_id}，无法计算退款金额。"

    amount = float(order["amount"])
    coupon = float(order.get("coupon") or 0.0)
    freight = float(order.get("freight") or 0.0)

    if quality_issue:
        refund = round(amount + freight, 2)
        reason = f"质量问题，按平台政策退全款（含运费 ¥{freight}）"
    else:
        refund = round(amount - coupon, 2)
        reason = f"非质量问题，退回实付金额（订单 ¥{amount} - 优惠券 ¥{coupon}）"

    return (
        f"[计算结果] 订单 {order_id}（{order['product']}）应退 ¥{refund}。"
        f"计算依据：{reason}。退款原路返回，审核通过后 1-3 个工作日到账。"
    )
