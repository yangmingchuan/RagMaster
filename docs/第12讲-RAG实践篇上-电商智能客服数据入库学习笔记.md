# RAG 实践篇（上）学习笔记 | 电商智能客服的数据入库

> 本篇主题：用 Next.js、LangChain.js 和 Milvus 搭建电商智能客服 RAG 项目，并完成数据准备、数据导入、文本切块、Embedding 和向量存储。

## 学习目标

这一篇开始从理论进入项目实战。

学完需要掌握：

1. 为什么选择“电商智能客服”作为 RAG 实战场景。
2. 一个 RAG 应用的前端、后端和核心引擎如何分层。
3. 如何准备结构化和非结构化数据。
4. 如何把本地文档加载、切块、向量化并存入向量数据库。
5. 如何用本地 HuggingFace 模型做 Embedding。

一句话概括：

> 实践篇（上）主要完成 RAG 系统的“索引阶段”：把电商商品和售后政策数据处理成可检索的向量知识库。

## 一、为什么选电商智能客服

RAG 最适合垂直领域知识问答，而电商客服是很典型的场景。

它同时包含两类数据：

| 数据类型 | 示例 | 特点 |
| --- | --- | --- |
| 结构化数据 | 商品列表、价格、库存、参数 | 字段固定，答案要准确 |
| 非结构化数据 | 售后政策、用户协议、服务说明 | 长文本，需要语义理解 |

电商客服里常见问题：

```text
有降噪耳机吗？
这个商品还有库存吗？
七天无理由退货怎么处理？
耳机保修多久？
```

这些问题既可能查商品表，也可能查政策文档，所以很适合练习多源数据 RAG。

## 二、项目整体架构

项目基于 Next.js 全栈架构。

技术栈：

| 层 | 技术 |
| --- | --- |
| 前端 | Next.js、React、Tailwind CSS |
| 后端 API | Next.js API Route |
| RAG 框架 | LangChain.js |
| 向量数据库 | Milvus，失败时降级 MemoryVectorStore |
| Embedding | 本地 `Xenova/bge-small-zh-v1.5` |
| LLM | OpenAI 兼容接口模型 |

核心目录：

```text
rag/
  demo-data/
    products.csv
    refund_policy.md
  src/
    app/
      api/
        chat/
        rag-demo/
      page.tsx
      rag-demo/
    lib/
      rag.ts
```

目录职责：

| 路径 | 作用 |
| --- | --- |
| `demo-data/` | 存放原始数据 |
| `src/app/page.tsx` | 客服聊天页面 |
| `src/app/api/chat/route.ts` | 用户聊天接口 |
| `src/app/api/rag-demo/` | RAG 流程调试接口 |
| `src/lib/rag.ts` | RAG 核心引擎 |

## 三、用户问题的处理流程

以用户问“有降噪耳机吗？”为例：

```text
前端输入问题
  ↓
请求 api/chat/route.ts
  ↓
调用 RAGEngine.chat(query)
  ↓
将问题向量化
  ↓
在 Milvus 中检索相关商品或政策片段
  ↓
对检索结果进行筛选或重排序
  ↓
把相关片段和问题组装成 Prompt
  ↓
LLM 生成回答
  ↓
返回前端展示
```

这一篇主要实现前半段：

```text
数据准备 -> 文档加载 -> 文本切块 -> Embedding -> 向量存储
```

## 四、数据准备

项目准备两份典型数据。

### 4.1 商品数据

文件：

```text
demo-data/products.csv
```

用途：

- 模拟商城库存表。
- 表示结构化数据。
- 每一行可以变成一个商品 Document。

CSV 的优势：

```text
表头能告诉模型每个字段的含义。
例如 1299 是价格，而不是型号。
```

### 4.2 售后政策数据

文件：

```text
demo-data/refund_policy.md
```

用途：

- 模拟售后、退货、保修政策。
- 表示非结构化长文本。

Markdown 的优势：

```text
# 和 ## 天然表示文档层级。
后续切块时可以尽量保留完整政策段落。
```

## 五、RAGEngine 的核心职责

项目把 RAG 逻辑集中在：

```text
src/lib/rag.ts
```

核心类：

```ts
class RAGEngine {
  private vectorStore;
  private embeddings;
  private llm;
  private isMemoryStore;
}
```

职责：

| 属性/方法 | 作用 |
| --- | --- |
| `vectorStore` | 保存 Milvus 或 MemoryVectorStore |
| `embeddings` | 文本向量化模型 |
| `llm` | 调用大模型生成回答 |
| `isMemoryStore` | 标记是否降级为内存向量库 |
| `init()` | 初始化向量库连接 |
| `addDocument()` | 加载、切分、向量化并存储文档 |
| `getRAGEngine()` | 单例导出，避免重复初始化 |

为什么用单例：

```text
Embedding 模型和向量库初始化成本高。
开发环境热重载时，如果重复初始化，会浪费内存和时间。
```

## 六、初始化逻辑

`init()` 方法的核心逻辑：

```text
尝试连接 Milvus
  ↓
连接成功：使用 Milvus
  ↓
连接失败：降级为 MemoryVectorStore
```

这样设计的好处：

- Milvus 可用时，使用持久化向量库。
- Milvus 不可用时，系统仍能用内存向量库跑通流程。
- 开发阶段容错更高。

要点：

```text
Milvus 是主方案。
MemoryVectorStore 是兜底方案。
```

## 七、数据导入流程

`addDocument(fileBuffer, fileName)` 负责完整 ETL 流程。

```text
Load 加载
  ↓
Split 切块
  ↓
Embed 向量化
  ↓
Store 存储
```

### 7.1 Load：按文件类型选择 Loader

| 文件类型 | 处理方式 |
| --- | --- |
| `.csv` | 使用 `CSVLoader` |
| `.pdf` | 优先 `WebPDFLoader`，失败后回退 `PDFLoader` |
| `.txt` / `.md` | 按普通文本读取 |

CSV 处理结果：

```text
每一行商品数据 -> 一个 Document
pageContent: 商品字段和值
metadata: source、line 等来源信息
```

Markdown 处理结果：

```text
整篇政策文本 -> Document
后续再由 Splitter 切成 Chunk
```

PDF 处理思路：

```text
先用 WebPDFLoader 保留较好的阅读顺序。
如果失败，再用基础 PDFLoader。
```

## 八、文本切块

文档加载后，需要切成 Chunk。

使用：

```ts
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 800,
  chunkOverlap: 100,
  separators: ["\n\n", "\n", "。", "！", "？"],
});
```

参数解释：

| 参数 | 含义 |
| --- | --- |
| `chunkSize: 800` | 每个 Chunk 约 800 字符 |
| `chunkOverlap: 100` | 相邻 Chunk 重叠 100 字符 |
| `separators` | 优先按段落、换行、中文句号等切分 |

为什么是 800：

```text
bge-small-zh 模型适合 512 到 1024 token 左右的输入。
800 是一个折中值。
太短会语义破碎，太长会影响检索效果。
```

## 九、本地 Embedding 模型

项目使用本地模型：

```text
Xenova/bge-small-zh-v1.5
```

选择本地 Embedding 的原因：

- 不依赖远程 API。
- 没有按次调用成本。
- 数据不需要发送到外部 Embedding 服务。
- 中文语义检索效果较好。

## 十、自定义 LocalHuggingFaceEmbeddings

文章中没有直接使用 LangChain 默认的 HuggingFace Embedding，而是自定义了一个 Embeddings 类。

原因：

```text
默认 HuggingFaceTransformersEmbeddings 可能会联网下载模型。
国内环境容易超时。
自定义后可以强制从本地 models 目录加载。
```

自定义 Embeddings 需要实现两个方法：

| 方法 | 作用 |
| --- | --- |
| `embedDocuments(texts)` | 批量把文档 Chunk 转成向量 |
| `embedQuery(text)` | 把用户问题转成向量 |

核心模型加载逻辑：

```text
pipeline("feature-extraction", model, {
  cache_dir: 本地模型目录,
  quantized: true
})
```

关键参数：

| 参数 | 作用 |
| --- | --- |
| `feature-extraction` | 文本特征提取，也就是生成向量 |
| `cache_dir` | 指定本地模型缓存目录 |
| `quantized: true` | 使用量化模型，减少内存占用 |
| `pooling: "mean"` | 对 token 向量取平均，得到句向量 |
| `normalize: true` | 向量归一化，方便余弦相似度计算 |

## 十一、下载模型到本地

本地模型放在：

```text
models/
```

下载思路：

```text
使用 @xenova/transformers 的 pipeline
把 remoteHost 指向 HuggingFace 镜像
允许远程下载
指定 cache_dir 为 ./models
```

执行完成后，后续 Embedding 就能从本地模型目录加载。

注意：

- 如果找不到 `@xenova/transformers`，通常是依赖没装或执行目录不对。
- 下载模型只需要做一次。
- 生产环境建议把模型目录纳入部署资源管理。

## 十二、向量化与混合存储

文本切块后进入向量化和存储。

核心逻辑：

```text
如果 vectorStore 不存在：
  创建向量库并写入文档
否则：
  追加文档到现有向量库
```

存储策略：

```text
优先 Milvus
失败则 MemoryVectorStore
```

### 12.1 创建向量库

流程：

```text
尝试 Milvus.fromDocuments(splitDocs, embeddings, config)
  ↓
成功：文档写入 Milvus
  ↓
失败：切换 isMemoryStore = true
  ↓
使用 MemoryVectorStore.fromDocuments(splitDocs, embeddings)
```

### 12.2 追加文档

如果已有向量库：

```text
vectorStore.addDocuments(splitDocs)
```

这样后续上传的新商品表、新政策文档，都可以继续追加进知识库。

## 十三、本篇完成了什么

实践篇（上）完成的是 RAG 索引阶段。

```text
项目初始化
  ↓
准备商品数据和售后政策
  ↓
实现 RAGEngine
  ↓
加载 CSV / PDF / Markdown
  ↓
递归文本切块
  ↓
使用本地 bge-small-zh 向量化
  ↓
优先写入 Milvus，失败回退内存向量库
```

也就是说，到这里为止，系统已经具备：

- 接收本地数据文件。
- 解析不同格式文档。
- 切成适合检索的 Chunk。
- 生成语义向量。
- 写入向量数据库。

但还没完整展开：

- 用户查询检索。
- 检索结果重排序。
- Prompt 组装。
- 最终问答生成。

这些通常会放在实践篇（下）继续实现。

## 十四、关键实现点速查

| 模块 | 关键点 |
| --- | --- |
| 项目架构 | Next.js 全栈，前端、API、RAG 引擎在一个项目 |
| 数据类型 | CSV 商品表 + Markdown 售后政策 |
| RAG 引擎 | `src/lib/rag.ts` 中统一管理 |
| 向量库 | 优先 Milvus，失败回退 MemoryVectorStore |
| Embedding | 本地 `Xenova/bge-small-zh-v1.5` |
| 切块 | `RecursiveCharacterTextSplitter` |
| Chunk 参数 | `chunkSize: 800`，`chunkOverlap: 100` |
| Loader | CSV、PDF、普通文本按类型处理 |
| 单例 | 用 `getRAGEngine()` 防止重复初始化 |

## 十五、复习检查

1. 电商智能客服为什么适合做 RAG 实战？
2. 商品 CSV 和售后 Markdown 分别代表哪类数据？
3. `RAGEngine` 为什么要做成单例？
4. 为什么初始化时要支持 Milvus 失败后回退内存向量库？
5. `addDocument()` 的 ETL 流程包括哪几步？
6. 为什么使用 `RecursiveCharacterTextSplitter`？
7. 为什么要自定义 `LocalHuggingFaceEmbeddings`？
8. `embedDocuments` 和 `embedQuery` 有什么区别？
9. 为什么向量要 `normalize: true`？
10. 本篇完成的是 RAG 的哪个阶段？

## 小结

RAG 实践篇（上）的核心是：

```text
把电商数据变成可被语义检索的向量知识库。
```

完整理解：

1. 商品数据和售后政策分别代表结构化与非结构化知识。
2. `RAGEngine` 统一负责加载、切块、向量化和存储。
3. 本地 Embedding 模型降低成本，也减少外部依赖。
4. Milvus 提供持久化向量存储，MemoryVectorStore 提供开发兜底。
5. 本篇打通了索引阶段，为后续检索和回答生成做准备。
