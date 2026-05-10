# Study Agent

基于 LangGraph 的智能学习助手，通过 RAG（检索增强生成）实现对教材、论文等学术文档的深度问答。

## 架构概览

```
PDF/Markdown 文档
       │
       ▼
 ┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
 │  MinerU API │────▶│  md_loader   │────▶│   md_splitter    │
 │  (PDF→MD)   │     │  (解析/分块)  │     │  (四层分块)       │
 └─────────────┘     └──────────────┘     └────────┬─────────┘
                                                    │
                                                    ▼
                     ┌──────────────┐     ┌──────────────────┐
                     │   Retriever  │◀────│  data_embedding  │
                     │  (混合检索)   │     │  (向量化→Milvus)  │
                     └──────┬───────┘     └──────────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   LLM Agent  │
                     │  (问答生成)   │
                     └──────────────┘
```

## 四层分块体系

文档被解析为四个层级的 chunk，各司其职：

| 层级 | 角色 | 内容 | 用途 |
|------|------|------|------|
| **L1** 章 | 索引节点 | 仅标题 | 层级导航、溯源 |
| **L2** 小节 | 索引节点 | 仅标题 | 层级导航、溯源 |
| **L3** 知识块 | 主召回层 | 完整语义段落（≥1500字符） | 混合检索的主要目标 |
| **L4** 证据块 | 细粒度层 | 短段落（≤400字符） | 精确匹配、关键词命中 |

检索时仅搜索 L3/L4，命中后通过 `parent_chunk_id` / `root_chunk_id` 回溯 L1→L2→L3 全链路上下文。

### Chunk ID 设计

采用扁平序号格式，支持 Agent 通过 ±1 获取相邻 chunk：

```
{doc_prefix}_L1_0000   ← 第1章
{doc_prefix}_L1_0001   ← 第2章
{doc_prefix}_L2_0000   ← 第1个小节
{doc_prefix}_L3_0000   ← 第1个知识块
{doc_prefix}_L3_0001   ← 第2个知识块（相邻）
{doc_prefix}_L4_0000   ← 第1个证据块
```

## 项目结构

```
src/agent/
├── rag/
│   ├── Loader/
│   │   ├── base_loader.py        # 加载器基类
│   │   ├── md_loader.py          # Markdown 解析器（TOC提取、标题分类、分块、页码映射）
│   │   ├── doc_loader.py         # PDF/Word/Excel 加载器（旧版）
│   │   └── minerU.py             # MinerU API 集成（PDF→Markdown）
│   ├── splitter/
│   │   ├── base_splitter.py      # 分块器基类
│   │   ├── md_splitter.py        # Markdown 四层分块器
│   │   └── text_splitter.py      # 纯文本三层分块器（旧版）
│   ├── data_embedding.py         # 入库管线：chunk → embedding → Milvus
│   ├── milvus_manage.py          # Milvus 客户端（schema、索引、CRUD）
│   └── retriever.py              # 检索管线：混合检索 → 多层上下文回溯
├── models/
│   ├── chat_model.py             # 对话模型配置
│   └── embedding_model.py        # Embedding 模型配置
├── context/
│   └── prompt_builder.py         # Prompt 构建器
├── schema/
│   ├── persona_state.py          # Agent 状态定义
│   └── tools_schema.py           # 工具 Schema
├── tools/
│   └── web_tool.py               # Web 搜索工具
├── utils/
│   ├── logger_handler.py         # 日志工具
│   └── path_handler.py           # 路径工具
├── agent.py                      # Agent 主入口
└── graph.py                      # LangGraph 图定义
```

## 核心管线

### 1. 文档入库

```python
from agent.rag.data_embedding import RAGPipelineService

service = RAGPipelineService(model_name="dashscope", dimensions=1024)
prepared = service.ingest_file(r"data\算法基础\算法基础.md")
service.close()
```

入库流程：

1. **PDF 预处理**（可选）：MinerU API 将 PDF 转为 Markdown + 图片 + `content_list.json`
2. **md_loader.load()**：解析 TOC、分类标题层级、提取结构化 section + block、映射页码
3. **md_splitter.split()**：四层分块，L3 使用段落合并（≥1500字符）+ 上下文衔接
4. **data_embedding**：批量向量化 → 写入 Milvus（去重检查 + MD5 跳过已处理文档）

### 2. 检索问答

```python
from agent.rag.retriever import Retriever

retriever = Retriever(model_name="dashscope", dimensions=1024)
contexts = retriever.retrieve("快速排序的时间复杂度是多少？", limit=5)
```

检索流程：

1. **Query Rewriting**：LLM 改写查询，增加语义关键词
2. **Hybrid Search**：Dense（语义向量）+ BM25（稀疏关键词）双路检索，仅搜 L3/L4
3. **Rerank**：结果重排序（预留 BGE-Reranker 接口）
4. **Multi-layer Context**：命中 chunk → 回溯 L1/L2/L3 祖先节点，组装完整上下文

### Milvus Schema（15 字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT64 (PK) | 自增主键 |
| `text_dense` | FLOAT_VECTOR | 语义向量 |
| `text_sparse` | SPARSE_FLOAT_VECTOR | BM25 稀疏向量 |
| `text` | VARCHAR(16384) | chunk 原文 |
| `doc_id` | VARCHAR | 文档唯一 ID（内容指纹） |
| `filename` | VARCHAR | 文件名 |
| `file_path` | VARCHAR | 文件路径 |
| `chunk_id` | VARCHAR | chunk 唯一 ID |
| `parent_chunk_id` | VARCHAR | 父 chunk ID |
| `root_chunk_id` | VARCHAR | 根 chunk ID（L1） |
| `chunk_level` | INT64 | 层级（1-4） |
| `title_path` | VARCHAR | 标题路径（如 `第1章 / 1.1 节 / 1.1.1`） |
| `title` | VARCHAR | 当前标题 |
| `content_type` | VARCHAR | 内容类型（chapter/section/...） |
| `page_number` | INT64 | 页码 |

## 环境配置

### 依赖

```bash
conda create -n openai python=3.12
conda activate openai
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

# LangSmith（可选）
LANGSMITH_API_KEY=lsv2...
```

## 快速开始

### 启动 LangGraph Server

```bash
langgraph dev
```

### 手动入库文档

```python
from agent.rag.Loader.minerU import MinerUParser
from agent.rag.data_embedding import RAGPipelineService

# PDF → Markdown
parser = MinerUParser()
output_dir = parser.parse(r"data\教材.pdf")

# Markdown → Milvus
service = RAGPipelineService(model_name="dashscope")
service.ingest_file(r"data\教材\教材.md")
service.close()
```

## 开发

- LangGraph Studio 支持热重载，修改代码后自动生效
- 可在 Studio 中编辑历史状态并从任意节点重新运行
- 集成 [LangSmith](https://smith.langchain.com/) 进行链路追踪和性能分析

## License

MIT
