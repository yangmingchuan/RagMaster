# RagMaster RAG Demo

这是一个本地可运行的 Next.js + LangChain RAG Demo，参考 `ai-engineer-practice/rag` 的课程项目实现，并扩展成电商智能客服场景。

项目包含两个主要页面：

- `/`：电商智能客服端到端体验，支持上传资料、知识库入库、聊天检索、查看引用来源。
- `/rag-demo`：RAG 原理分步演示，覆盖数据加载、文本切分、Embedding、向量库、检索增强和生成。

## 当前已实现

- 文件上传：支持 PDF、TXT、CSV、Markdown。
- 文档加载：使用 LangChain `CSVLoader`、`PDFLoader`、`WebPDFLoader` 和普通文本加载。
- 文本切分：使用 `RecursiveCharacterTextSplitter`，并保留 source、line、loc 等元数据。
- 本地 Embedding：使用本地 `Xenova/bge-small-zh-v1.5` 模型，通过 `@xenova/transformers` 生成 512 维向量。
- 向量库：
  - 默认优先连接 Milvus。
  - Milvus 不可用时自动回退到 `MemoryVectorStore`。
- 检索：基于向量相似度召回 Top K 文档片段。
- Rerank 与生成：
  - 配置 `DEEPSEEK_API_KEY` 后执行 LLM Rerank 和最终回答生成。
  - 未配置 key 时仍可完成真实 Embedding 检索，并返回引用来源。
- 知识库管理：支持查看 chunk 列表、查看聊天引用来源、重置知识库。
- RAG 原理演示：`/rag-demo` 提供 Load、Split、Embedding、Memory/Chroma、Retrieval、MultiQuery、HyDE、Contextual Compression、Parent Document、Generation 等示例。

```text
上传资料 -> 文档加载 -> 文本切块 -> Embedding -> 向量入库 -> 检索召回 -> LLM Rerank -> Prompt 生成 -> 返回答案与引用
```

## 环境变量

先复制示例 key 文件：

```bash
cp .env.local.example .env.local
```

`.env.local` 里的关键配置可以先留空：

```bash
# Milvus 地址。不启动 Milvus 时，系统会自动回退到 MemoryVectorStore。
MILVUS_ADDRESS=localhost:19530

# 主页面智能客服使用。留空时不会执行 LLM Rerank 和最终生成。
DEEPSEEK_API_KEY=

# /rag-demo 页面里的 OpenAI Embedding、MultiQuery、HyDE、Generation 示例使用。
OPENAI_API_KEY=

# 预留给 Cohere Rerank 示例或后续扩展。当前主流程没有强依赖。
COHERE_API_KEY=
```

## 安装与启动

如果系统 PATH 里没有 `node` / `npm`，可以使用本机 Node 路径：

```bash
export PATH=/Users/mingchuanyang/.local/node-v22/bin:$PATH
```

安装依赖：

```bash
cd /Users/mingchuanyang/Downloads/GitHubWorkSpaces/RagMaster/ragdemo
npm install
```

启动 Next.js：

```bash
npm run dev
```

打开：

```text
http://localhost:4012
```

## 启动 Milvus

项目支持 Milvus Lite，本地已有启动脚本：

```bash
python3 scripts/start_milvus.py
```

默认监听：

```text
localhost:19530
```

如果不启动 Milvus，项目会自动使用内存向量库。内存模式适合快速调试，但进程重启后知识库会清空。

## 推荐测试数据

可直接上传：

- `demo-data/products.csv`
- `demo-data/refund_policy.md`
- `demo-data/sample.txt`
- `demo-data/sample.pdf`

推荐提问：

- 推荐一款降噪耳机
- 人体工学椅 V2 多少钱？
- 七天无理由退货的条件是什么？
- 数码产品保修期多久？

## 常用脚本

```bash
npm run dev          # 启动开发服务，端口 4012
npm run build        # 构建生产版本
npm run start        # 启动生产版本，端口 4012
npm run lint         # ESLint 检查
npm run db:up        # 启动 Milvus Lite
npm run db:down      # 停止 Milvus 进程
```

## 接口说明

- `POST /api/upload`：上传文件并写入知识库。
- `POST /api/chat`：基于知识库进行 RAG 问答。
- `GET /api/documents`：分页查看知识库 chunks。
- `POST /api/reset`：重置知识库。
- `POST /api/rag-demo`：运行 `/rag-demo` 页面里的分步 RAG 示例。

## 注意

- `models/Xenova/bge-small-zh-v1.5` 是本地 Embedding 模型目录，代码默认禁止联网下载模型。
- 没有 `DEEPSEEK_API_KEY` 时，主聊天流程会停在“真实向量检索 + sources 返回”，不会生成最终 LLM 答案。
- `/rag-demo` 中涉及 OpenAI Embedding 或 LLM 的按钮需要 `OPENAI_API_KEY`，也可以在页面输入框临时填写。
- `lint` 当前还存在一些 TypeScript/ESLint 规范项需要整理，但 `npm run build` 已可通过。
