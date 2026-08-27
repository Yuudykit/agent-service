# -*- coding: utf-8 -*-
"""
embedding 后端对比实验 —— 证明选型价值（面试高光材料）

对比 hash（特征哈希，零依赖） vs bge（bge-small-zh-v1.5，512 维）：
    同一批口语化 query，看 top1 检索是否正确命中对应 FAQ。

对抗实验（--no-synonyms）：
    禁用 retriever 的人工同义词归一化表（_SYNONYMS）。这是"剥离人工兜底"
    的压测：hash 靠 keywords/synonyms 才能全对，bge 靠语义本身就分得开。
    结果预期：hash 掉命中（漂移重现），bge 仍稳 —— 这就是选 embedding
    的量化依据（面试讲"为什么不用特征哈希"时直接引用这组数字）。

背景：
    特征哈希是 bigram 近似，没有语义概念——"邮费"、"没有收到"这类
    换说法会被漂移到不相关条目（实测过：→ 发票/缺货）。换成真实
    embedding 后，语义相近的句子聚到一起，漂移消失。

用法：
    python scripts/compare_embedders.py [--rebuild] [--model bge|bge-m3] [--no-synonyms]

依赖：sentence-transformers（pip install sentence-transformers）
HuggingFace 下载慢时：set HF_ENDPOINT=https://hf-mirror.com
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 这些 query 曾在 hash 后端下检索漂移 / 弱命中（评测反哺的失败案例）
PROBE_QUERIES = [
    "退换无忧也要自己拿邮费吗",      # 曾 → kb_020 发票（正确：kb_012 退货运费）
    "没有收到货",                    # 曾 → kb_002 缺货（正确：kb_007 丢件）
    "可以帮我转快递吗",              # 曾 → kb_046 商品参数（正确：kb_041 转人工）
    "这一款手机什么时候能够发货呀?",  # 正确：kb_001 发货时效
    "钱什么时候退回来",              # 正确：kb_010 退款到账（hash 下弱）
    "我退货的话上门取件收费吗",       # 正确：kb_012 退货运费
    "发票金额是怎么算的",            # 正确：kb_019 发票金额
]

# query → 期望命中的 kb id（人工标注 ground truth）
EXPECTED = {
    "退换无忧也要自己拿邮费吗": "kb_012",
    "没有收到货": "kb_007",
    "可以帮我转快递吗": "kb_041",
    "这一款手机什么时候能够发货呀?": "kb_001",
    "钱什么时候退回来": "kb_010",
    "我退货的话上门取件收费吗": "kb_012",
    "发票金额是怎么算的": "kb_019",
}


def run_backend(backend: str, rebuild: bool, no_synonyms: bool = False,
                pure_vector: bool = False) -> dict:
    """切到指定后端，重建索引（可选），对全部 probe query 跑 top1 检索。"""
    os.environ["EMBEDDER_BACKEND"] = backend
    import rag.build as build_mod
    import rag.retriever as retriever

    # 重新加载 embedder 与索引（单例缓存 + 索引缓存都要重置）
    import rag.embedder as embedder_mod
    embedder_mod._embedder = None
    retriever.reset_index_cache()

    if no_synonyms:
        # 对抗实验：剥离人工归一化表，暴露后端真实的检索能力
        retriever._SYNONYMS = []
    if pure_vector:
        # 纯向量压测：关键词通道权重归零，top1 完全由 embedding 决定
        retriever._WEIGHT_KW = 0.0

    if rebuild:
        build_mod.build_index()

    results = {}
    for q in PROBE_QUERIES:
        top1 = retriever.retrieve(q, top_k=1)[0]
        results[q] = {
            "hit": top1["id"],
            "expected": EXPECTED[q],
            "correct": top1["id"] == EXPECTED[q],
            "match_type": top1["match_type"],
            "question": top1["question"],
        }
    return results


def print_report(backend: str, results: dict) -> None:
    correct = sum(1 for r in results.values() if r["correct"])
    print(f"\n===== 后端: {backend} —— 命中 {correct}/{len(results)} =====")
    for q, r in results.items():
        mark = "✓" if r["correct"] else "✗"
        print(f"  {mark} {q[:20]:<22} → {r['hit']} {r['question'][:20]}  [{r['match_type']}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="跑对比前重建对应后端索引")
    parser.add_argument("--model", default="bge", choices=["bge", "bge-m3", "m3e"],
                        help="本地 ML 后端（默认 bge=bge-small-zh-v1.5）")
    parser.add_argument("--no-synonyms", action="store_true",
                        help="对抗实验：禁用同义词归一化表，剥离人工兜底")
    parser.add_argument("--pure-vector", action="store_true",
                        help="纯向量压测：关键词通道权重归零，top1 完全由 embedding 决定")
    args = parser.parse_args()

    tag = []
    if args.no_synonyms:
        tag.append("禁用同义词归一化")
    if args.pure_vector:
        tag.append("纯向量")
    tag = "（" + " + ".join(tag) + "）" if tag else ""

    hash_results = run_backend("hash", rebuild=False, no_synonyms=args.no_synonyms,
                               pure_vector=args.pure_vector)
    print_report(f"hash（特征哈希 512 维）{tag}", hash_results)

    ml_results = run_backend(args.model, rebuild=args.rebuild, no_synonyms=args.no_synonyms,
                             pure_vector=args.pure_vector)
    print_report(f"{args.model}（本地 embedding）{tag}", ml_results)

    # 汇总
    h = sum(1 for r in hash_results.values() if r["correct"])
    m = sum(1 for r in ml_results.values() if r["correct"])
    print(f"\n===== 总结：hash {h}/{len(hash_results)} vs {args.model} {m}/{len(ml_results)} =====")


if __name__ == "__main__":
    main()
