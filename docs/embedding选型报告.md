# Embedding 模型选型报告（与 DeepSeek 搭配）

模型信息、榜单和价格为调研日期的记录。第六节是项目内 7 条样本的对比实验。

> 调研日期：2026-08-27 ｜ 结论先行：**DeepSeek 官方没有 embedding 接口**，
> RAG 里 embedding 与 LLM 解耦是业界标准做法——DeepSeek 负责生成，
> embedding 选第三方专用模型（本地开源首选 BGE 系列，云端 API 首选阿里 text-embedding-v4）。
> **本机实测已跑通**：bge-small-zh-v1.5 通过 ModelScope 下载并落地为
> `EMBEDDER_BACKEND=bge`（见第六节压测数据），全量 58 评测通过。

---

## 一、为什么"适合 ds 的 embedding"是个伪命题

**事实（2026-08 实测确认）**：
- DeepSeek 官方 API 的 OpenAI 兼容面**只有 Chat Completions**（chat/reasoning），
  **没有 /embeddings 端点**，也没有 Assistants/Files。
- 网上流传的"deepseek-embedding"接口来自第三方镜像站，非官方文档，不可信。
- 用 DeepSeek 系列大模型（如 R1-Distill）自托管做 embedding 是 llama.cpp 的
  离题用法——推理型模型没有专门训练过向量表征，质量不如专用 embedding 模型，
  而且 7B 级模型做向量化性价比极差。

**结论**：不存在"DeepSeek 官方 embedding"。正确姿势是**分工**：
```
用户 query ──→ 专用 embedding 模型 ──→ 向量检索（召回）──→ DeepSeek（生成答案）
```
选型只看三件事：中文效果（C-MTEB）、部署成本（本地/API）、维度兼容性。

---

## 二、候选模型横向对比（数据来自 C-MTEB / MMTEB 榜单与社区实测）

| 模型 | 来源 | 参数量 | 维度 | 中文 MTEB | 部署 | 成本 | 备注 |
|------|------|--------|------|-----------|------|------|------|
| **BGE-M3** | BAAI 智源 | 567M | 1024 | **64.8**（榜首） | 本地开源 | 免费 | 100+ 语言、8192 长文本、dense+sparse+colbert 三检索信号，RAG 生产首选 |
| **bge-large-zh-v1.5** | BAAI 智源 | 335M | 1024 | 63.2 | 本地开源 | 免费 | 中文单语榜第一梯队；Top-5 检索 78%（200 条文档实测） |
| **bge-small-zh-v1.5** | BAAI 智源 | 24-118M | 512 | 61.8 | 本地开源 | 免费 | 轻量，CPU 可跑；**512 维与本项目 hash 后端维度一致，索引零改动** |
| m3e-base | MokaAI | 220M | 768 | 60.5 | 本地开源 | 免费 | 中文老牌"小钢炮"，35ms 低延迟 |
| gte-large-zh | 阿里达摩院 | 335M | 1024 | 59.2 | 本地开源 | 免费 | 阿里系，中文优化 |
| Qwen3-Embedding-0.6B | 阿里 | 600M | 1024 | 强 | 本地/API | 免费/付费 | 2025 最新，多语言强 |
| text-embedding-v4 | 阿里云 | — | 可调(64-2048) | 强 | **API** | 0.6 元/百万 token | 100+ 语种，OpenAI 兼容，免费额度 100 万 token/90 天 |
| embedding-3 | 智谱 | — | 1024 | 强 | API | 0.5 元/百万 token | 国内合规 |
| text-embedding-3-small/large | OpenAI | — | 1536/3072 | 中等（58.5/62.5） | API | $0.02/0.13 每百万 | 中文被高估，贵且需网络条件 |

**关键洞察**：
1. 中文场景开源模型（BGE）全面优于 OpenAI API——训练数据决定一切。
2. 567M 的 BGE-M3 在中文检索上击败 7B+ 通用大模型（准确率 89% vs 72%）——
   专业分工 > 堆参数。
3. 维度影响存储与内存：3072 维（OpenAI large）比 1024 维多占 3 倍。

---

## 三、本机环境适配（RTX 5080 16GB）

| 方案 | 显存占用 | 推理 | 结论 |
|------|---------|------|------|
| bge-small-zh-v1.5 | ~250MB | CPU 毫秒级 | ✅ **首选**（512 维兼容现有索引） |
| BGE-M3 | ~1.2GB | GPU 流畅（5080 富余） | ✅ 追求极致检索效果时切它 |
| dashscope API | 0 | 网络调用 | ✅ 不想跑本地 / 演示云端方案时 |

---

## 四、推荐结论（分层）

1. **本地首选：bge-small-zh-v1.5** —— 免费、离线、数据不出域、512 维与
   项目 hash 后端维度一致（`kb_index_hash.json` → `kb_index_bge.json` 一键重建）。
2. **本地进阶：BGE-M3** —— 中文检索综合最强，面试展示"换 embedding 后
   检索漂移消失"的对比实验用它最有说服力；重建索引（512→1024 维）即可。
3. **云端备选：阿里 text-embedding-v4** —— 不想维护模型 / 需要更大知识库时，
   OpenAI 兼容协议 + 100 万 token 免费额度，直接换 base_url 接入。
4. **明确不选**：OpenAI embedding（中文弱+贵）、DeepSeek 蒸馏模型做 embedding
   （离题用法，质量差）。

---

## 五、落地方式（本项目）

```
EMBEDDER_BACKEND=hash      特征哈希向量（默认，零依赖）
EMBEDDER_BACKEND=bge       bge-small-zh-v1.5（512 维，推荐）
EMBEDDER_BACKEND=bge-m3    bge-m3（1024 维，重建索引后使用）
EMBEDDER_BACKEND=m3e       m3e-small（早期预留）
EMBEDDER_BACKEND=dashscope 阿里 text-embedding-v4（需 DASHSCOPE_API_KEY）
```

- 索引文件按后端隔离：`kb_index_{backend}.json`，切换后端不会互相覆盖。
- 检索接口 `retrieve(query)` 完全不变——embedding 是插件化的，换模型零业务改动。
- 本地后端依赖可通过 `python -m pip install -r requirements-embeddings.txt` 安装。
- 模型下载：国内环境优先 ModelScope（`snapshot_download('BAAI/bge-small-zh-v1.5')`，
  落到 `models/modelscope/models/`，`embedder.py` 自动优先加载本地；
  不依赖 ModelScope 时回落 HF，配 `HF_ENDPOINT=https://hf-mirror.com`。
  坑：transformers 5.x 的 xet 传输协议 HF 镜像不支持，需 `HF_HUB_DISABLE_XET=1`）。

---

## 六、实测对比（2026-08-27 本机跑通，`scripts/compare_embedders.py`）

7 条曾在 hash 后端下检索漂移/弱命中的真实口语 query，两个后端 × 三档压测：

| 压测档位 | hash（特征哈希） | bge-small-zh-v1.5 |
|---------|----------------|-------------------|
| 默认（混合 + 同义词归一化） | 7/7 | 7/7 |
| 禁用同义词归一化（剥人工兜底） | 5/7 | 5/7 |
| **纯向量裸考**（syn 关 + 关键词权重归零） | **4/7** | **5/7** |

**三档压测的意义**（面试直接引用）：
1. 默认档两边都满分——**人工兜底（synonyms + keywords）已经把 hash 的坑填平了**，
   说明"评测反哺数据"是有效的；但 hash 的命中 5/7 靠 keyword 通道硬撑（match_type
   清一色 `[keyword]`），而 bge 的命中全是 `[both]`——向量通道本身就对，关键词只是双保险。
2. 禁同义词后 hash 5/7、bge 5/7，**但失败方式暴露本质差异**：
   - hash 漂到 kb_046「商品详细参数」——与"转快递"**语义无关**，bigram 近似崩塌；
   - bge 漂到 kb_006「修改配送地址」——**语义相关**（都在快递主题里），只是没到最精确那条。
   - "没有收到货"两边都错（→缺货）——这个 query 本身有歧义（缺货 vs 丢件），
     纯向量/纯关键词都治不了，**这正是混合检索 + 人工规则该兜的边界**。
3. 纯向量裸考：bge 5/7 > hash 4/7，且 hash 新错漂到 kb_020「开票」、kb_047「保质期」
   ——完全无关；bge 的新错仍是语义相邻（退款主题）。**语义天花板高一个档次**。

**结论**：embedding 的价值不是"替换人工兜底"，而是**把检索的天花板抬高**——
人工规则兜得住 7/7，但那是"修出来的正确"；bge 让正确"自带"，规则只处理
真正的歧义边界（如"转快递""没有收到"）。混合检索 = 语义上限 + 规则兜底。

---

