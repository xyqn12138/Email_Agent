# Study Agent

基于 LangGraph 的个人超级知识库 Agent，通过多阶段 RAG 管线实现对教材、课件等学术文档的深度问答。

## 架构概览

```
PDF / Markdown 文档
        │
        ▼
 ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
 │  MinerU API  │────▶│  md_loader   │────▶│   md_splitter    │
 │  (PDF → MD)  │     │  (结构解析)   │     │  (四层分块)       │
 └──────────────┘     └──────────────┘     └────────┬─────────┘
                                                     │
                                                     ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
  │   Retriever  │◀────│  Reranker    │◀────│  data_embedding  │
  │  (混合检索)   │     │  (qwen3)     │     │  (向量化→Milvus)  │
  └──────┬───────┘     └──────────────┘     └──────────────────┘
         │
         ▼
  ┌──────────────┐
  │ LangGraph    │     ┌──────────────────────────────────────┐
  │ ReAct Agent  │────▶│ tools: search / context / image / web│
  └──────────────┘     └──────────────────────────────────────┘
```

## 四层分块体系

| 层级 | 角色 | 内容 | 用途 |
|------|------|------|------|
| **L1** 章 | 索引节点 | 仅标题 | 层级导航、溯源 |
| **L2** 小节 | 索引节点 | 仅标题 | 双阶段检索锚点 |
| **L3** 知识块 | 主召回层 | 语义段落（~1500字符） | 混合检索主要目标 |
| **L4** 证据块 | 细粒度层 | 短段落（~400字符） | 精确匹配、关键词命中 |

### Chunk ID 设计

扁平序号格式，支持 Agent 通过 ±N 获取相邻 chunk：

```
{doc_prefix}_L1_0000   ← 第1章
{doc_prefix}_L2_0000   ← 第1个小节
{doc_prefix}_L3_0000   ← 第1个知识块
{doc_prefix}_L3_0001   ← 第2个知识块（相邻）
{doc_prefix}_L4_0000   ← 第1个证据块
```

## 检索管线

### 智能路由

根据查询特征自动选择最优检索路径：

```
用户查询
   │
   ├─ HyDE 开启 → LLM 生成假想文档 → 直接 L3/L4 检索
   │
   ├─ 短查询（≤15字）→ 双阶段检索
   │     └─ Stage 1: L2 粗定位 → Stage 2: 范围限定 L3/L4 细召回
   │
   └─ 长查询 → 直接 L3/L4 全局检索
```

### 检索流程

1. **Query Rewriting**（可选）：LLM 改写查询，增加语义关键词
2. **HyDE**（可选）：LLM 生成假想答案文本，用其向量检索，适合口语化查询
3. **Hybrid Search**：Dense（语义向量）+ BM25（稀疏关键词）双路检索，RRF 融合
4. **Rerank**：qwen3-rerank 精排，top10 召回 → top3 精选
5. **Multi-layer Context**：命中 chunk → 回溯 L1/L2/L3 祖先节点，组装完整上下文

### 质量自适应

RAG Tool 支持 `mode=auto`：当 rerank 置信度低于阈值时，自动升级为 HyDE 模式重检索。

### 图片处理

入库时正则剥离 Markdown 图片语法（`![](path)`）后再向量化，原路径存入 `image_paths` 字段，Agent 可通过 `view_image` 工具按需查看。

## Agent 工具

| 工具 | 说明 | 关键参数 |
|------|------|----------|
| `knowledge_base_search` | 知识库检索 | `query`, `mode`(auto/standard/hyde), `limit`, `advanced` |
| `fetch_neighbor_context` | 获取相邻 chunk 上下文 | `chunk_id`, `n_before`, `n_after` |
| `view_image` | 查看知识库引用的图片 | `image_path` |
| `web_search` | 互联网搜索（智谱 API） | `query` |

## 项目结构

```
src/agent/
├── rag/
│   ├── Loader/
│   │   ├── base_loader.py        # 加载器基类
│   │   ├── md_loader.py          # Markdown 解析（TOC 提取、标题分类、分块、页码映射）
│   │   └── minerU.py             # MinerU API（PDF → Markdown）
│   ├── splitter/
│   │   ├── base_splitter.py      # 分块器基类
│   │   └── md_splitter.py        # Markdown 四层分块器
│   ├── data_embedding.py         # 入库管线：图片剥离 → chunk → embedding → Milvus
│   ├── milvus_manage.py          # Milvus 客户端（schema、索引、CRUD、邻居查询）
│   └── retriever.py              # 检索管线：路由 → 混合检索 → 重排序 → 多层上下文
├── models/
│   ├── chat_model.py             # 对话模型（多 Provider 注册）
│   ├── embedding_model.py        # Embedding 模型（DashScope / OpenAI / 本地）
│   └── reranker_model.py         # 重排序模型（qwen3-rerank via DashScope）
├── tools/
│   ├── rag_tool.py               # 知识库检索工具（自适应质量升级）
│   ├── context_tool.py           # 邻居上下文工具
│   ├── image_tool.py             # 图片查看工具
│   └── web_tool.py               # Web 搜索工具
├── utils/
│   ├── logger_handler.py         # 日志工具
│   └── path_handler.py           # 路径工具
├── agent.py                      # 交互式对话入口
└── graph.py                      # LangGraph Agent 定义
```

## Milvus Schema（16 字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT64 (PK) | 自增主键 |
| `text_dense` | FLOAT_VECTOR | 语义向量 |
| `text_sparse` | SPARSE_FLOAT_VECTOR | BM25 稀疏向量（自动由 BM25 Function 生成） |
| `text` | VARCHAR(16384) | chunk 清洁文本（已剥离图片语法） |
| `doc_id` | VARCHAR | 文档唯一 ID（内容指纹） |
| `filename` | VARCHAR | 文件名 |
| `file_path` | VARCHAR | 文件路径 |
| `chunk_id` | VARCHAR | chunk 唯一 ID |
| `parent_chunk_id` | VARCHAR | 父 chunk ID |
| `root_chunk_id` | VARCHAR | 根 chunk ID（L1） |
| `chunk_level` | INT64 | 层级（1-4） |
| `title_path` | VARCHAR | 标题路径 |
| `title` | VARCHAR | 当前标题 |
| `content_type` | VARCHAR | 内容类型（chapter/section/...） |
| `page_number` | INT64 | 页码 |
| `image_paths` | VARCHAR(4096) | 图片路径（分号分隔） |

## 环境配置

### 依赖

```bash
conda create -n study python=3.12
conda activate study
pip install -e . "langgraph-cli[inmem]"
```

### 环境变量

创建 `.env` 文件：

```env
# MinerU（PDF 解析）
MINERU_API_KEY=your_mineru_api_key
MINERU_BASE_URL=https://mineru.net

# Milvus
MILVUS_URI=http://localhost:19530
MILVUS_HOST=localhost
MILVUS_PORT=19530
COLLECTION_NAME=study_agent

# 模型
DASHSCOPE_API_KEY=your_dashscope_key

# 检索参数（可选，均有默认值）
RETRIEVE_SHORT_QUERY_THRESHOLD=15    # 短查询阈值（字数）
RETRIEVE_HYBRID_LIMIT=15             # 混合检索召回数
RERANK_TOP_N=5                       # 重排序精选数
RERANK_SCORE_THRESHOLD=0.3           # 质量阈值（低于则触发重检索）
RERANK_MODEL=qwen3-rerank            # 重排序模型

# LangSmith（可选）
LANGSMITH_API_KEY=lsv2_...
```

## 快速开始

### 启动 Agent

```bash
# 交互式对话
python -m agent.agent

# 启动 LangGraph Server（Studio 可视化调试）
langgraph dev
```

### 入库文档

```python
from agent.rag.data_embedding import RAGPipelineService

# Markdown 直接入库
service = RAGPipelineService(model_name="dashscope", dimensions=1024)
service.ingest_file(r"data\算法基础\算法基础.md")

# PDF 入库（自动 MinerU 转换）
service.ingest_file(r"data\教材.pdf")
service.close()
```

### 单独测试检索

```python
from agent.rag.retriever import Retriever

retriever = Retriever(model_name="dashscope", dimensions=1024)
contexts = retriever.retrieve("快速排序的时间复杂度是多少？", limit=3)
for ctx in contexts:
    print(ctx["title_path"], ctx.get("rerank_score"))
```

## 开发

- LangGraph Studio 支持热重载，修改代码后自动生效
- 集成 [LangSmith](https://smith.langchain.com/) 进行链路追踪和性能分析

## License

MIT
