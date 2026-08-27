# agent-service · 手写智能体客服

> 零框架依赖，全部从零手写的智能体客服系统。每个模块直接对标真实 JD 的核心技术要求。
> 数据来自 **JDDC 京东对话挑战赛**（1 万段真实客服会话提炼出 47 条 FAQ 知识库）。

## 技术对标

| 模块 | JD 技术要求 | 状态 |
|------|------------|------|
| `core/agent_loop.py` | Function Calling + 工具调用 + ReAct 循环 | ✅ 完成 |
| `core/context.py` | 上下文管理（窗口/截断/摘要占位） | ✅ 完成 |
| `core/memory.py` | Agent Memory（摘要 + 用户画像 + 持久化） | ✅ 完成 |
| `core/reasoning.py` | ReAct 规划与推理（显式 Thought/Action/Observation） | ✅ 完成 |
| `core/prompts.py` | 分层 Prompt 工程（角色/技能/规则/few-shot/记忆层） | ✅ 完成 |
| `rag/` | RAG 管线（chunker → embedder → build → retriever → reranker） | ✅ 完成 |
| `tools/` | 多工具设计（registry 自动 schema + 4 个业务工具） | ✅ 完成 |
| `eval/` | Agent 评测（58 用例四指标 + 19 项单元测试） | ✅ 完成 |

## 架构

```
用户输入
   │
   ▼
core/agent_loop.py ──► core/reasoning.py（ReAct: Think→Act→Observe）
   │                        │
   │                        ▼
   │                 tools/registry.py（装饰器注册 + 签名自动推导 schema）
   │                        │
   │                        ├── search_kb.py ──► rag/（混合检索：向量+关键词→重排）
   │                        ├── query_info.py（订单结构化查询）
   │                        ├── calculate_refund.py（退款金额计算）
   │                        └── fallback.py（兜底/转人工，不编造原则）
   │
   ├── core/context.py（消息窗口/截断/摘要）
   ├── core/memory.py（对话记忆：历史摘要 + 用户画像）
   └── core/prompts.py（分层系统提示词，技能层随工具表动态生成）

core/agent_loop.py 输出最终回答
```

## 快速开始

```bash
# 0. 首次使用：复制配置模板并填入你的 DeepSeek API key（真实模式才需要，mock 可跳过）
cp config.example.json config.json

# 1. 离线演示（不调 API，关键词路由模拟模型判断，零依赖可跑）：
python main.py --mock --verbose --memory

# 2. Web 版（浏览器聊天页：SSE 流式输出 / 会话持久化 / 转人工交接包）：
python webapp/server.py            # 默认 mock 模式，勾掉前端开关切真实模式

# 3. 跑正式评测（58 用例 = 28 基础 + 30 多轮，四指标报告）：
python -m eval.eval_script

# 4. 跑单元测试（19 项）：
pytest tests/

# 5. 重建 RAG 索引（改过 data/kb.json 后执行）：
python -m rag.build
```

> CI：push 后 GitHub Actions 自动跑单测 + mock 评测（零 API 成本），
> 见 `.github/workflows/ci.yml`。

## 生产化特性（P1/P2）

| 特性 | 说明 |
|------|------|
| SSE 流式输出 | 真实模式逐字渲染，工具调用过程实时可见（`⚙️ 查询订单信息…`） |
| 会话持久化 | 每轮落盘 `data/sessions/`，服务重启后原会话不失忆 |
| API 健壮性 | 60s 超时 + 429/5xx 指数退避重试 2 次 + 失败降级话术（不甩裸异常） |
| 转人工交接包 | `data/handoffs/` 落盘工单（诉求/订单号/话题/情绪/最近轮次），纯规则抽取零成本 |
| 用量护栏 | 日 token 预算 + 会话轮次上限，超限零成本熔断（不调 API） |
| 结构化日志 | `logs/events.jsonl` 一轮一行（工具链/token/延迟/降级标记） |
| bad case 收集 | 降级/转人工/兜底自动追加 `data/bad_cases.jsonl`，作为知识库迭代原料 |

## 部署：最后一步（填入 API key）

> 所有离线环节已全部完成。**真实模式已验证通过**（见下方验证结果）。

### 环境准备（conda 虚拟环境）

```bash
# 创建环境（注意：若 D:\Anaconda\envs 有写保护，conda 会默认建到用户目录）
conda create -n agent-service python=3.12 -y
conda activate agent-service
pip install openai numpy
# 若 conda 报 "Upload did not complete / JSONDecodeError"，用：
#   CONDA_NO_PLUGINS=true CONDA_PKGS_DIRS=D:/tmp/conda_pkgs conda create -n agent-service python=3.12 -y --solver=classic
```

### 真实模式启动

1. 打开 `config.json`，填入你的 DeepSeek API key：
   ```json
   { "api_key": "sk-你的key", "base_url": "https://api.deepseek.com", "model": "deepseek-chat" }
   ```
2. 启动真实模式：`python main.py --verbose`
3. 可选：RAG 换语义更强的 embedding（推荐）——装 `sentence-transformers` 后
   设 `EMBEDDER_BACKEND=bge` 切到 bge-small-zh-v1.5（512 维，实测 58/58 评测通过，
   选型依据见 `docs/embedding选型报告.md`）；默认 `hash` 是零依赖特征哈希，
   `bge-m3` / `dashscope` 见报告第五节

### 真实模式验证结果（2026-08-24）

```bash
conda run -n agent-service python scripts/run_real_eval.py   # 全量 5 问
conda run -n agent-service python scripts/run_real_eval.py --only C1   # 单用例（省 token）
```

| 用例 | 问题 | 结果 |
|------|------|------|
| C1 退款时效 | 退款多久能到账？ | ✅ 1-3 工作日 + 银行 3-5 工作日 |
| C2 多轮发票→税号 | 开发票 → 税号填错怎么改 | ✅✅ 正确承接"刚才那个订单"，提醒已开发票不可改 |
| C3 天气兜底 | 今天上海天气？ | ✅ 明确服务边界 |
| C4 防幻觉 | 你们老板姓什么？ | ✅ 拒绝回答，不编造 |
| C5 订单查询 | JD20260815001 到哪了 | ✅ 已发货/已揽收/预计 8-18 送达 |

消耗：12 次 API 调用 / 18,793 tokens（单次约 1.4k-2.1k，system prompt+工具 schema 占大头）。
`run_real_eval.py` 内置 usage 拦截统计；多轮用例共享会话、其余独立会话控制上下文膨胀。

**说明**：mock 模式的关键词路由只用于离线演示与测试；真实模式下
agent_loop 走 OpenAI 兼容协议（tools=tools_schema），模型自主决定调用哪个工具。

## 数据资产（来自 JDDC 真实客服数据）

| 文件 | 内容 |
|------|------|
| `data/raw/jddc_chat.txt` | JDDC 原始对话（20.8MB，212,808 行，10,943 个会话）**不入库**，`scripts/build_kb_from_jddc.py` 可从 JDDC 数据集重建 |
| `data/kb.json` | **47 条 FAQ 知识库**（11 个主题，从真实对话提炼，含 40 条 variants 问法增强） |
| `data/kb_index_{hash,bge}.json` | RAG 向量索引（按 embedding 后端分文件，`rag/build.py` 生成，**不入库**，首次运行自动重建） |
| `data/orders.json` | 模拟订单数据（query_info / calculate_refund 用） |
| `data/eval_set.json` | 28 个基础评测用例（含 2 个手写多轮：工具/兜底/防幻觉） |
| `data/eval_multiturn.json` | 30 个多轮评测用例（120 个真实会话自动反哺生成；合计 58 用例、多轮 32） |
| `data/dialogues.jsonl` | 120 个真实完整客服会话（带 topic/turn_count 等元数据） |
| `scripts/` | 数据管线（analyze → build_kb → sample → enrich → multiturn）+ 开发期验收脚本 |

## 进度

- [x] Day 4：Agent 主循环 + 上下文管理（P0 闭环）
- [x] Day 5：RAG 管线（混合检索 + 重排）
- [x] Day 6：Memory（摘要/画像/持久化）+ ReAct 显式推理
- [x] Day 7：工具集扩充（query_info + calculate_refund）+ Prompt 完善
- [x] Day 8：评测（28/28）+ 单元测试（19/19）+ 部署文档
- [x] 数据管线重构：parse_raw 统一口径 + variants 增强，多轮评测扩至 58 用例（hash/bge 双后端 58/58 全绿）
- [x] P1 可用性：SSE 流式 + 会话持久化 + API 重试降级（顺手修了最终回答不回填 messages 的存量 bug）
- [x] P2 业务真实感：转人工交接包 + token 日预算/轮次护栏 + JSONL 日志
- [x] P3 质量回路：GitHub Actions CI + bad case 自动收集
