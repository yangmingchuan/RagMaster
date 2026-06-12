# RAG 学习与应用指南

> 基于 RagMaster 项目（Next.js + LangChain）的 RAG 实战教学文档

---

## 目录

1. [什么是 RAG](#1-什么是-rag)
2. [RAG 核心流程](#2-rag-核心流程)
3. [项目架构总览](#3-项目架构总览)
4. [分步详解](#4-分步详解)
   - [4.1 文档加载（Load）](#41-文档加载load)
   - [4.2 文本切分（Split）](#42-文本切分split)
   - [4.3 向量化（Embedding）](#43-向量化embedding)
   - [4.4 向量存储（Vector Store）](#44-向量存储vector-store)
   - [4.5 检索召回（Retrieval）](#45-检索召回retrieval)
   - [4.6 重排序（Rerank）](#46-重排序rerank)
   - [4.7 提示词构建与生成（Prompt & Generation）](#47-提示词构建与生成prompt--generation)
5. [高级检索技术](#5-高级检索技术)
6. [动手实践](#6-动手实践)
7. [常见问题](#7-常见问题)

---

## 1. 什么是 RAG

**RAG（Retrieval-Augmented Generation，检索增强生成）** 是一种将信息检索与大语言模型（LLM）结合的技术架构。

### 为什么需要 RAG？

大语言模型有两大天然缺陷：

| 问题 | 说明 |
|------|------|
| **知识截止** | 模型训练数据有截止日期，不知道训练后发生的事情 |
| **幻觉** | 模型可能编造不存在的事实，尤其在私有领域 |

RAG 的解决思路：**让 LLM 回答问题前，先从外部知识库中检索相关信息，再基于检索到的真实资料生成答案。**

```
用户提问 → 在知识库中检索相关文档 → 将文档+问题一起发给LLM → LLM基于文档生成答案
```

### 类比理解

把 LLM 想象成一个闭卷考试的学生——他只能靠记忆作答。RAG 就是给他一本参考书，让他**翻书找到相关段落，再基于书里的内容作答**。答案有了依据，不再凭空瞎编。

---

## 2. RAG 核心流程

一个完整的 RAG 系统分为**离线入库**和**在线问答**两个阶段：

```
┌─────────────────────────────────────────────────────┐
│                   离线入库阶段                        │
│                                                      │
│  PDF/CSV/TXT → 文档加载 → 文本切块 → Embedding向量化  │
│                                          ↓            │
│                                    向量数据库存储      │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                   在线问答阶段                        │
│                                                      │
│  用户提问 → Query向量化 → 向量相似度检索 → 召回Top K  │
│                                                    ↓  │
│                        LLM Rerank重排序 → 取Top 3     │
│                                        ↓              │
│                          构建Prompt → LLM生成 → 答案  │
└─────────────────────────────────────────────────────┘
```

本项目完整实现了上述 **7 个环节**，下面逐一详解。

---

## 3. 项目架构总览

### 技术栈

| 层级 | 技术选型 |
|------|---------|
| 框架 | Next.js 16 + React 19 + TypeScript |
| RAG 框架 | LangChain |
| 本地 Embedding | Xenova/bge-small-zh-v1.5（512 维，中文优化） |
| 向量数据库 | Milvus（优先）/ MemoryVectorStore（回退） |
| LLM | DeepSeek-V3（兼容 OpenAI 协议） |
| 样式 | Tailwind CSS 4 |

### 目录结构

```
ragdemo/
├── src/
│   ├── app/
│   │   ├── page.tsx                  # 主页：智能客服聊天
│   │   ├── rag-demo/page.tsx         # /rag-demo：RAG原理分步演示
│   │   └── api/
│   │       ├── chat/route.ts         # POST /api/chat — 问答接口
│   │       ├── upload/route.ts       # POST /api/upload — 文件上传
│   │       ├── reset/route.ts        # POST /api/reset — 重置知识库
│   │       ├── documents/route.ts    # GET /api/documents — 查看chunks
│   │       └── rag-demo/             # RAG各环节独立演示API
│   └── lib/
│       └── rag.ts                    # ⭐ 核心：RAGEngine 类（全部逻辑）
├── models/Xenova/bge-small-zh-v1.5/  # 本地Embedding模型
├── demo-data/                        # 测试数据
└── scripts/start_milvus.py           # Milvus启动脚本
```

### 核心类：RAGEngine

所有 RAG 逻辑集中在 `src/lib/rag.ts` 的 `RAGEngine` 类中，约 820 行。对外暴露的 API 非常简洁：

```typescript
const rag = await getRAGEngine();         // 获取单例
await rag.addDocument(buffer, fileName);  // 上传文档 → 全流程入库
const { answer, sources } = await rag.chat(query);  // 问答
await rag.reset();                        // 重置知识库
await rag.getDocuments(page, pageSize);   // 分页查看chunks
```

---

## 4. 分步详解

### 4.1 文档加载（Load）

**做什么**：把用户上传的文件（PDF/CSV/TXT/Markdown）解析成 LangChain 的 `Document` 对象。

**代码位置**：`src/lib/rag.ts` → `RAGEngine.addDocument()` 方法

```typescript
// 根据文件扩展名选择不同的 Loader
if (fileName.endsWith(".pdf")) {
  // PDF：先尝试 WebPDFLoader（基于 pdfjs-dist，表格/多栏效果好）
  // 失败则回退到 PDFLoader（基于 pdf-parse）
  const loader = new WebPDFLoader(blob);
  docs = await loader.load();
} else if (fileName.endsWith(".csv")) {
  // CSV：使用 CSVLoader，自动解析表头
  const loader = new CSVLoader(blob);
  docs = await loader.load();
} else {
  // 纯文本：直接读取为 Document
  docs = [new Document({ pageContent: text, metadata: { source: fileName } })];
}
```

**关键概念**：
- `Document` 对象包含 `pageContent`（文本内容）和 `metadata`（元数据，如来源文件名、页码等）
- PDF 加载有两种 Loader，WebPDFLoader 对复杂排版更友好
- 加载后的 Document 可能很大（整本 PDF 几十页），需要下一步切分

---

### 4.2 文本切分（Split）

**做什么**：将大段文档切成小块（chunks），每个 chunk 是检索的最小单位。

**为什么需要切分？**
1. Embedding 模型有输入长度限制（bge-small-zh-v1.5 是 512 token）
2. 检索时小块更精准，不会把无关内容也带进来
3. 多个小块可以拼成 LLM 的上下文窗口

**代码位置**：`src/lib/rag.ts` → `addDocument()` 方法

```typescript
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 800,    // 每块最多 800 字符
  chunkOverlap: 100, // 相邻块重叠 100 字符（保持语义连贯）
  separators: ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
  // 分隔符优先级：先按段落切，再按句子切，最后按词切
});

const splitDocs = await splitter.splitDocuments(docs);
```

**切分方式对比**（在 `/rag-demo` 页面可体验）：

| 切分器 | 原理 | 适用场景 |
|--------|------|---------|
| RecursiveCharacterTextSplitter | 按优先级递归切分 | **通用推荐** |
| CharacterTextSplitter | 按固定分隔符切 | 格式规整的文本 |
| TokenTextSplitter | 按 token 数切 | 精确控制 LLM 输入长度 |

**chunkSize 和 chunkOverlap 怎么选？**
- chunkSize 太小 → 信息碎片化，丢失上下文
- chunkSize 太大 → 检索精度下降，噪音多
- chunkOverlap 太小 → 关键信息可能被切断在边界
- 本项目 800/100 是经验值，在中文电商场景下效果较好

---

### 4.3 向量化（Embedding）

**做什么**：将文本转换成数学向量（一组数字），使语义相近的文本在向量空间中距离也相近。

**为什么需要 Embedding？**
计算机无法直接理解"语义相似"，但可以计算向量之间的余弦距离。两段意思相近的文字，它们的向量在高维空间中会很接近。

**本项目方案**：使用本地 `bge-small-zh-v1.5` 模型，无需联网调用 API

**代码位置**：`src/lib/rag.ts` → `LocalHuggingFaceEmbeddings` 类

```typescript
// 自定义 Embeddings 类，继承 LangChain 的 Embeddings 基类
class LocalHuggingFaceEmbeddings extends Embeddings {
  // 初始化 pipeline（只执行一次）
  async _initPipeline() {
    this.pipeline = await pipeline("feature-extraction", this.model, {
      cache_dir: this.cacheDir,  // 指定本地模型目录
      quantized: true,           // 量化加速，减少内存占用
    });
  }

  // 文档 → 向量
  async embedDocuments(documents: string[]): Promise<number[][]> {
    for (const doc of documents) {
      const output = await this.pipeline(doc, {
        pooling: "mean",    // 对所有token向量求平均 → 句向量
        normalize: true,    // 归一化，方便计算余弦相似度
      });
      results.push(Array.from(output.data));
    }
    return results;  // 每个文档返回一个 512 维向量
  }

  // 用户 Query → 向量（同上逻辑，只是单个输入）
  async embedQuery(query: string): Promise<number[]> { ... }
}
```

**关键配置**：
- `env.localModelPath`：指向项目根目录的 `models/` 文件夹
- `env.allowRemoteModels = false`：禁止联网下载，强制使用本地模型
- `quantized: true`：模型量化为 int8，内存从 ~400MB 降到 ~100MB
- 输出维度：512 维（bge-small-zh-v1.5 的默认维度）

**为什么用本地模型而不是 OpenAI Embedding？**
| | 本地 bge-small-zh-v1.5 | OpenAI text-embedding-3-small |
|---|---|---|
| 费用 | 免费 | 按 token 计费 |
| 延迟 | 首次加载 ~2s，后续毫秒级 | 网络往返 ~200ms |
| 中文效果 | 专门为中文优化 | 多语言通用 |
| 隐私 | 数据不离开本机 | 数据发送到 OpenAI |

---

### 4.4 向量存储（Vector Store）

**做什么**：将向量化后的 chunks 存储起来，并支持高效的相似度搜索。

**本项目方案**：双模策略 — 优先连接 Milvus，不可用时自动回退到 MemoryVectorStore

#### Milvus（生产级）

Milvus 是一个开源向量数据库，专门为大规模向量检索设计。

当你运行 `python3 scripts/start_milvus.py` 时，会在本地启动一个 Milvus Lite 实例。

**代码位置**：`src/lib/rag.ts` → `addToMilvusStore()`

```typescript
// 创建集合（类似关系数据库的建表）
await client.createCollection({
  collection_name: "rag_collection",
  fields: [
    { name: "langchain_primaryid", data_type: Int64, is_primary_key: true, autoID: true },
    { name: "langchain_text",      data_type: VarChar, max_length: 8192 },
    { name: "source",              data_type: VarChar, max_length: 1024 },
    { name: "loc",                 data_type: VarChar, max_length: 2048 },
    { name: "langchain_vector",    data_type: FloatVector, dim: 512 },
  ],
  index_params: {
    index_type: "FLAT",      // 精确搜索（数据量小时用 FLAT）
    metric_type: "COSINE",   // 余弦相似度
  },
});
```

**为什么用 Milvus？**
- 数据持久化，重启不丢失
- 支持百万级向量检索
- 支持多种索引类型（FLAT/IVF_FLAT/HNSW 等）

#### MemoryVectorStore（开发/回退）

当 Milvus 不可用时，自动使用内存向量库：

```typescript
// 纯内存，适合快速开发和调试
this.vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  this.embeddings  // 传入我们的 Embeddings 实例
);

// 检索时直接调用
const results = await this.vectorStore.similaritySearchWithScore(query, 10);
```

**MemoryVectorStore vs Milvus**：

| | MemoryVectorStore | Milvus |
|---|---|---|
| 持久化 | 进程重启丢失 | 数据持久化 |
| 数据量 | 受内存限制 | 支持海量数据 |
| 部署 | 零配置 | 需启动服务 |
| 适用 | 开发调试 | 生产环境 |

---

### 4.5 检索召回（Retrieval）

**做什么**：将用户问题向量化，在向量库中找到最相似的 chunks。

**代码位置**：`src/lib/rag.ts` → `RAGEngine.chat()` 方法

```typescript
// 1. 向量检索 — 召回 Top 10（比最终需要的多，给 Rerank 留候选）
const initialResults = await this.vectorStore.similaritySearchWithScore(query, 10);
// 返回值：[Document, score][]  — score 是余弦相似度

// 每个结果包含：
// - doc.pageContent：文本内容
// - doc.metadata.source：来源文件名
// - score：向量相似度分数
```

**关键参数 `k` 的选择**：
- 本项目召回 10 个，因为后续有 Rerank 会再筛选
- 如果没有 Rerank，直接取 3~5 个即可
- 召回太少 → 可能漏掉相关信息
- 召回太多 → 后续 LLM 处理的上下文变长，成本增加

---

### 4.6 重排序（Rerank）

**做什么**：向量相似度 ≠ 语义相关性。Rerank 用 LLM 对召回的 10 个候选重新打分排序，选出真正相关的 3 个。

**为什么需要 Rerank？**
向量检索只看 embedding 的几何距离，但对于 LLM 最终生成答案来说，"这段文字能否回答用户问题"才是关键。Rerank 就是让 LLM 来当裁判，筛选出真正有用的资料。

**代码位置**：`src/lib/rag.ts` → `RAGEngine.chat()`

```typescript
// 批量模式：将 10 个候选一次性发给 LLM 打分
const rerankPrompt = `
你是一个文档相关性评分专家。
请判断以下文档片段与用户问题的相关性，并给出 0-10 的评分。

用户问题: "${query}"

候选文档列表:
[文档ID: 0]
内容: ${candidatesText[0]}

------------------------

[文档ID: 1]
内容: ${candidatesText[1]}
...
`;

// LLM 返回 JSON 数组
// [{"id": 0, "score": 9.5}, {"id": 1, "score": 3}, ...]

// 过滤：只保留 score >= 6 的文档
// 排序：按 score 降序
// 截断：取 Top 3
```

**评分阈值为什么是 6？**
| 分数 | 含义 |
|------|------|
| 0-3 | 完全不相关，丢弃 |
| 4-5 | 弱相关，可能提及但不直接回答，丢弃 |
| 6-7 | 相关，可以辅助回答 |
| 8-10 | 高度相关，直接回答用户问题 |

阈值设太低 → 噪音多；设太高 → 可能漏掉有用信息。6 是经验平衡值。

**降级策略**：如果 LLM Rerank 失败（JSON 解析错误等），自动回退到取原始向量检索的 Top 3。

---

### 4.7 提示词构建与生成（Prompt & Generation）

**做什么**：将检索到的 chunks 拼入 Prompt，让 LLM 基于真实资料生成答案。

**代码位置**：`src/lib/rag.ts` → `RAGEngine.chat()`

```typescript
const template = `
<角色>
你是一名"睿智商城"的智能客服。请基于以下提供的商品信息或售后政策回答用户的问题。

<限制>
不要编造事实。如果信息中没有提到相关内容，请礼貌地引导用户转人工服务。

<用户问题>
{question}

<已知信息>
{context}
`;

// 拼接上下文
const contextStr = retrievedDocs.map((d) => d.pageContent).join("\n\n");

// 构建完整 Prompt
const formattedPrompt = await prompt.format({ question: query, context: contextStr });

// 发送给 LLM 生成答案
const chain = this.llm.pipe(new StringOutputParser());
const response = await chain.invoke(formattedPrompt);

return { answer: response, sources: retrievedDocs };  // 返回答案 + 引用来源
```

**Prompt 设计要点**：

1. **角色设定**：告诉 LLM 它是什么角色（电商客服），影响语气和回答风格
2. **限制条件**：明确禁止编造，不知道就承认 —— 这是 RAG 减少幻觉的关键
3. **格式分离**：用 `<角色>` `<限制>` `<已知信息>` 标签分隔不同部分，帮助 LLM 理解结构
4. **来源透明**：返回 `sources` 给前端，用户可以验证答案来自哪里

---

## 5. 高级检索技术

本项目在 `/rag-demo` 页面中实现了多种进阶检索技术（需要 `OPENAI_API_KEY`）：

### 5.1 Multi-Query（多角度查询）

**原理**：让 LLM 将用户的一个问题改写成多个不同角度的问题，分别检索后合并结果。

```
用户问："LangChain 怎么用？"
       ↓ LLM 改写
"LangChain 的基本用法是什么？"
"LangChain 有哪些核心组件？"
"LangChain 的快速入门教程"
       ↓ 分别检索
三组结果合并去重 → 更全面的召回
```

**适用场景**：用户问题太简短或模糊时，多角度查询能召回更多相关信息。

### 5.2 HyDE（假设文档嵌入）

**原理**：先让 LLM 根据问题生成一个"假设的答案"，再用这个答案去做向量检索。

```
用户问："LangChain 的作用是什么？"
       ↓ LLM 生成假设答案
"LangChain 是一个用于开发 LLM 应用的框架，提供了 Chains、Agents 等组件..."
       ↓ 用假设答案做向量检索
检索结果与假设答案语义更接近 → 召回更精准
```

**为什么有效？** 问题和答案的表述方式不同。用户的提问是疑问句，知识库中的文档是陈述句。用 LLM 生成的"假设答案"去检索，向量更接近真实文档的语义空间。

### 5.3 Contextual Compression（上下文压缩）

**原理**：检索到文档后，让 LLM 从中提取只与问题相关的句子，去掉无关噪音。

```
检索到的文档："LangChain 是 LLM 应用框架。它支持 Python 和 JS。
               安装方式是 pip install langchain。最新版本是 0.3。"
                                      ↓ 用户问的是安装方式
LLM 压缩后："安装方式是 pip install langchain。"
```

**适用场景**：文档 chunk 较大、包含多种信息时，压缩能提高最终答案的精度。

### 5.4 Parent Document Retriever（父文档检索）

**原理**：将文档切成大小两种粒度 —— 用小块做检索（精准），检索到后返回对应的大块（完整上下文）。

```
原始文档：一篇 5000 字的长文
    ↓ 父文档切分：每块 500 字符
父块 A │ 父块 B │ 父块 C │ ...
    ↓ 子文档切分：每块 100 字符
子块1 子块2 子块3 子块4 子块5 ...

检索时：用子块匹配 → 找到子块3 → 返回父块A（完整上下文）
```

**适用场景**：文档较长，小块精准匹配，大块保证上下文完整。

---

## 6. 动手实践

### 6.1 前置准备

```bash
# 1. 进入项目目录
cd ragdemo

# 2. 确保 Node.js 可用
export PATH=/Users/mingchuanyang/.local/node-v22/bin:$PATH

# 3. 安装依赖（首次）
npm install

# 4. 配置 API Key（可选，不配置也能体验检索环节）
cp .env.local.example .env.local
# 编辑 .env.local，填入 DEEPSEEK_API_KEY 和 OPENAI_API_KEY
```

### 6.2 启动项目

```bash
# 启动 Next.js 开发服务器
npm run dev
# 访问 http://localhost:4012
```

### 6.3 体验检索流程（无需 API Key）

1. 打开 http://localhost:4012
2. 点击文件选择框，上传 `demo-data/products.csv`
3. 看到 `Success: File processed successfully (XX chunks)` 即入库成功
4. 点击 **View Knowledge Base** 查看所有 chunks
5. 点击快捷问题 **"推荐一款降噪耳机"**
6. 尽管没有 LLM 答案，但可以看到 **View Sources** 按钮
7. 点击查看每个 source 的向量分数和 chunk 内容

这就是 RAG 检索环节的核心体验。

### 6.4 完整体验（需要 DEEPSEEK_API_KEY）

配置 Key 后重启，同样的问题会经历完整流程：

```
用户："推荐一款降噪耳机"
  → 向量检索：召回 10 个候选 chunks
  → LLM Rerank：10 个候选打分排序，取 Top 3
  → Prompt 构建：将 Top 3 chunks 拼入客服 Prompt
  → LLM 生成：基于真实商品数据回答，附带 View Sources
```

### 6.5 启动 Milvus（可选）

```bash
# 启动本地 Milvus Lite
python3 scripts/start_milvus.py
# 监听 localhost:19530

# 重启 Next.js，RAGEngine.init() 会自动连接 Milvus
```

不启动 Milvus 也能正常运行，系统会自动回退到 MemoryVectorStore。

### 6.6 体验 RAG 原理分步演示

访问 http://localhost:4012/rag-demo（需要 OPENAI_API_KEY）

点击各按钮观察每一步的输出：
- Policy Loader → 加载 refund_policy.md
- Product Loader → 加载 products.csv
- Text Splitters → 对比三种切分方式的结果
- OpenAI Embedding → 查看向量维度
- Memory Store → 存入内存向量库
- Basic Retrieval → 基础检索
- Multi-Query / HyDE / Compression / Rerank / Parent Document → 高级检索
- Basic Generation → 最终答案生成

---

## 7. 常见问题

### Q: MemoryVectorStore 和 Milvus 什么时候切换？

系统启动时自动检测：如果能连上 Milvus 服务端，就用 Milvus；否则用 MemoryVectorStore。不需要手动配置。

### Q: 知识库的数据存在哪里？

- Milvus 模式：存在 `milvus_data/` 目录下，重启不丢失
- Memory 模式：只存在于 Node.js 进程内存中，重启即消失

### Q: 为什么上传 PDF 有时很慢？

PDF 解析是 CPU 密集型操作。如果 PDF 页数多或有复杂表格，WebPDFLoader 可能需要几秒钟。这是正常的。

### Q: 没有 API Key 能用到什么程度？

可以完整体验：文件上传 → 文本切分 → Embedding → 向量入库 → 向量检索 → 查看 Sources。只是缺少 LLM Rerank 和最终答案生成两个环节。

### Q: 如何更换 Embedding 模型？

修改 `src/lib/rag.ts` 中 `LocalHuggingFaceEmbeddings` 的 `model` 参数和 `modelsPath`。需要先将新模型下载到 `models/` 目录。注意不同模型的输出维度可能不同（会直接影响 Milvus 集合的 schema）。

### Q: chunkSize 怎么调优？

没有固定答案，取决于你的文档类型：
- 电商商品描述 → 800 左右合适
- 技术文档 → 可以设到 1000-1500，保留更完整的技术上下文
- 法律/合同 → 可以设小一点 300-500，因为每句话都可能很重要

建议先保持默认值试跑，观察检索效果后再微调。

---

## 总结

RAG 的本质是 **"先检索，再生成"**——让 LLM 回答问题时有了"参考资料"。本项目完整实现了 RAG 的 7 个核心环节，从文档上传到最终答案，每一步都可以独立观察和调试。

建议的学习路径：

1. 先不配置 API Key，体验检索环节（上传 → 切分 → Embedding → 检索 → 查看 Sources）
2. 再配置 DEEPSEEK_API_KEY，体验完整流程（加上 Rerank 和 LLM 生成）
3. 打开 `/rag-demo` 页面，逐个运行高级检索技术的演示
4. 阅读 `src/lib/rag.ts` 源码，理解每个环节的具体实现
