# -*- coding: utf-8 -*-
"""
向量化 —— RAG 管线第 2 步

职责：把文本变成向量（embedding），供相似度计算。

为什么用 embedding（面试必答）:
    - 关键词匹配答不了"换个说法"的问题（"退款多久到账" vs "钱什么时候退回来"）
    - embedding 把语义相近的句子映射到相近的向量空间
    - 余弦相似度 = 语义相似度

后端（EMBEDDER_BACKEND 环境变量切换，接口统一为 encode）:
    1. hash（默认）：特征哈希向量（feature hashing）——零依赖、离线可跑
       把中文按"相邻两字"（bigram）打散成 512 维向量，面试可直接讲原理。
    2. bge（推荐）：sentence-transformers + BAAI/bge-small-zh-v1.5
       智源开源中文 embedding，512 维与 hash 后端一致，索引无需改维度。
       C-MTEB 中文榜第一梯队，纯本地、免费、数据不出域。
    3. bge-m3：sentence-transformers + BAAI/bge-m3
       中文 Embedding 综合最强（C-MTEB 64.8 榜首），1024 维，
       支持 dense + sparse + colbert 三种检索信号。本机 RTX 5080 可流畅推理。
    4. m3e：sentence-transformers + moka-ai/m3e-small（早期预留，兼容）
    5. dashscope：阿里云 text-embedding-v4 API（Qwen3-Embedding 系列）
       0.6 元/百万 token，OpenAI 兼容协议，需 DASHSCOPE_API_KEY + base_url。

选型结论（调研于 2026-08）：
    - DeepSeek 官方只有 chat/reasoning API，**没有 embedding 接口**——
      RAG 里"embedding 选型"与"LLM 选型"解耦，DeepSeek 负责生成，
      embedding 用第三方专用模型（这是业界标准做法）。
    - 中文检索效果：BGE-M3 > bge-large-zh-v1.5 > bge-small-zh-v1.5 > m3e-base。
    - 本项目数据量（47 条 FAQ）很小，bge-small-zh-v1.5 已足够；
      若换 bge-m3 需重建索引（维度 512 → 1024）。

为什么不用内置 hash()：Python 的 hash() 对 str 带随机种子（PYTHONHASHSEED），
同一进程不同次运行结果都不同，索引和查询会失配。必须用确定性哈希。
"""
from __future__ import annotations

import math
import os
import re
import zlib
from pathlib import Path

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

DIM = 512  # hash / bge-small 向量维度（bge-m3 为 1024，dashscope 可配）

# 后端名（EMBEDDER_BACKEND 取值）—— build.py 用它区分索引文件
BACKEND_HASH = "hash"
BACKEND_BGE = "bge"
BACKEND_BGE_M3 = "bge-m3"
BACKEND_M3E = "m3e"
BACKEND_DASHSCOPE = "dashscope"

# 各本地后端的 HuggingFace 模型名
_MODEL_NAMES = {
    BACKEND_BGE: "BAAI/bge-small-zh-v1.5",
    BACKEND_BGE_M3: "BAAI/bge-m3",
    BACKEND_M3E: "moka-ai/m3e-small",
}

# ModelScope 本地缓存根（国内下载模型的主通道，比 HF 镜像稳）。
# 基于模块文件位置解析，避免依赖"从项目根目录运行"这个前提。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_MODEL_ROOT = _PROJECT_ROOT / "models" / "modelscope" / "models"


def _resolve_model_path(model_name: str) -> str:
    """优先用 ModelScope 已下载的本地模型，否则原样返回 HF 模型名。

    ModelScope 的 snapshot_download 目录结构：
        models/modelscope/models/BAAI--bge-small-zh-v1.5/snapshots/master/
    本地命中 → sentence-transformers 离线加载，不依赖外网；
    本地没有 → 回落 HF 在线下载（可配 HF_ENDPOINT=https://hf-mirror.com）。
    """
    repo_dir = model_name.replace("/", "--")
    candidates = [
        _LOCAL_MODEL_ROOT / repo_dir / "snapshots" / "master",
        _LOCAL_MODEL_ROOT / repo_dir,
    ]
    for c in candidates:
        if (c / "config.json").exists():
            return str(c)
    return model_name

_SPACE_RE = re.compile(r"\s+")
# 单字功能词：先删掉再算 bigram，避免"的/了/我"这类高频噪声
_CHAR_STOPWORDS = set("的了吗呢啊吧呀哦噢嗯你我他她它们您咱请问想要能会是就很都也还不")
# 双字词：算完 bigram 后整体排除（"怎么"会被拆进多个 bigram 污染相似度）
_WORD_STOPWORDS = {
    "请问", "怎么", "什么", "多少", "多久", "哪里", "如何", "一下",
    "可以", "能不能", "是不是", "为什么", "想要", "需要", "这个", "那个",
    "我们", "你们", "他们", "一个", "可以", "的话",
}


def _bigrams(text: str) -> list[str]:
    """提取中文相邻两字（bigram），过滤停用词。

    "退款多久到账" → 退款/款多/久能/能到/到账（"多久"整体被排除）
    """
    t = _SPACE_RE.sub("", str(text))
    t = "".join(ch for ch in t if ch not in _CHAR_STOPWORDS)
    grams = [t[i : i + 2] for i in range(len(t) - 1)]
    return [g for g in grams if g not in _WORD_STOPWORDS and not any(w in g for w in _WORD_STOPWORDS if len(w) == 2)]


def _vector_from_bigrams(grams: list[str]) -> list[float]:
    """bigram 词频 → 特征哈希向量（TF 加权 + 随机符号 + L2 归一化）。"""
    freq: dict[str, int] = {}
    for g in grams:
        freq[g] = freq.get(g, 0) + 1

    vec = [0.0] * DIM
    for g, cnt in freq.items():
        # crc32 确定性哈希 → (位置, 符号)
        h = zlib.crc32(g.encode("utf-8"))
        idx = h % DIM
        sign = 1.0 if (h >> 16) & 1 else -1.0
        vec[idx] += sign * cnt

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class FeatureHashEmbedder:
    """零依赖向量化后端。接口与 sentence-transformers 的 encode 对齐。"""

    backend = BACKEND_HASH  # current_backend() 据此如实上报实际后端

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [_vector_from_bigrams(_bigrams(t)) for t in texts]


class SentenceTransformersEmbedder:
    """本地开源 embedding 后端：sentence-transformers + 指定模型。

    bge-small-zh-v1.5：512 维，与 hash 后端维度一致，切后端无需重建索引
    bge-m3：1024 维，中文检索综合最强，需重建索引
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入

        self.model = SentenceTransformer(_resolve_model_path(model_name))
        # sentence-transformers 6.x 改名：get_embedding_dimension（兼容旧版）
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = getter()

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


class DashScopeEmbedder:
    """云端 API 后端：阿里云 text-embedding-v4（Qwen3-Embedding 系列）。

    OpenAI 兼容协议，需环境变量：
        DASHSCOPE_API_KEY  阿里云百炼 API Key
        DASHSCOPE_BASE_URL 默认 https://dashscope.aliyuncs.com/compatible-mode/v1
    0.6 元/百万 token；支持动态维度（默认 1024，可调 512 与本地索引对齐）。
    """

    backend = BACKEND_DASHSCOPE

    def __init__(self, dimensions: int = 512) -> None:
        import openai  # 延迟导入

        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY 未设置。请到阿里云百炼控制台申请，"
                "或改用本地后端（EMBEDDER_BACKEND=bge）。"
            )
        base_url = os.environ.get(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.dim = dimensions

    def encode(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(
            model="text-embedding-v4",
            input=texts,
            dimensions=self.dim,
            encoding_format="float",
        )
        return [d.embedding for d in resp.data]


_embedder = None


def get_embedder():
    """加载嵌入模型（单例，常驻内存）。

    EMBEDDER_BACKEND 取值：
        hash      特征哈希向量（默认，零依赖离线可跑）
        bge       bge-small-zh-v1.5（512 维，推荐本地后端）
        bge-m3    bge-m3（1024 维，中文最强，需重建索引）
        m3e       m3e-small（早期预留）
        dashscope 阿里 text-embedding-v4 API（需 DASHSCOPE_API_KEY）
    本地 ML 后端失败时降级为特征哈希向量（现场演示不挂）。
    降级后 current_backend() 如实上报 "hash"——索引文件名跟着实际后端走，
    避免"kb_index_bge.json 里装着 hash 向量"的静默错配（查询向量与库
    向量来自不同后端，维度相同时余弦分数全是噪声且不报错）。
    """
    global _embedder
    if _embedder is None:
        backend = os.environ.get("EMBEDDER_BACKEND", BACKEND_HASH)
        if backend in _MODEL_NAMES:
            try:
                _embedder = SentenceTransformersEmbedder(_MODEL_NAMES[backend])
                _embedder.backend = backend
            except Exception as exc:  # pragma: no cover
                print(f"[embedder] {backend} 加载失败（{exc}），降级为特征哈希向量")
                _embedder = FeatureHashEmbedder()
        elif backend == BACKEND_DASHSCOPE:
            try:
                _embedder = DashScopeEmbedder()
            except Exception as exc:  # pragma: no cover
                print(f"[embedder] dashscope 不可用（{exc}），降级为特征哈希向量")
                _embedder = FeatureHashEmbedder()
        else:
            _embedder = FeatureHashEmbedder()
    return _embedder


def current_backend() -> str:
    """返回当前生效的后端名（build.py 用它区分索引文件）。

    单例已加载 → 报实际后端（降级后不再是环境变量声称的那个）；
    未加载 → 报环境变量指定的目标后端（首次构建索引前的路径计算）。
    """
    if _embedder is not None:
        return getattr(_embedder, "backend", BACKEND_HASH)
    return os.environ.get("EMBEDDER_BACKEND", BACKEND_HASH)


def encode(texts: list[str]) -> list[list[float]]:
    """把文本列表转成向量列表。"""
    return get_embedder().encode(texts)


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（向量已归一化 → 直接点积；未归一化则手动除模长）。"""
    if _HAS_NUMPY:
        na, nb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if na.size != nb.size or na.size == 0:
            return 0.0
        denom = (np.linalg.norm(na) * np.linalg.norm(nb)) or 1e-9
        return float(np.dot(na, nb) / denom)
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    denom = (na * nb) or 1e-9
    return dot / denom
