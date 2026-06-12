# 第 12 讲内容总结 | RAG 实战项目：电商智能客服

## 一句话总结

第 12 讲把前面几讲的 RAG 理论落到了一个完整项目里：用 Next.js + LangChain.js + Milvus 做一个电商智能客服，完成从数据入库、向量检索、LLM 重排序、Prompt 生成回答，到前端上传文件、查看知识库、聊天和展示引用来源的全流程。

## 本讲主线

```text
准备电商数据
  ↓
加载文档并切块
  ↓
本地 Embedding 向量化
  ↓
写入 Milvus 或 MemoryVectorStore
  ↓
提供上传、重置、查询、聊天 API
  ↓
前端完成上传、查看知识库、聊天问答
  ↓
返回答案并展示引用来源
```

这讲的核心不是单独学某个 API，而是理解一个 RAG 应用从“资料进入知识库”到“用户拿到可溯源回答”的完整闭环。

## 一、项目场景与架构

项目选择“电商智能客服”作为 RAG 场景，因为它同时包含两类典型知识：

| 数据类型 | 示例 | 价值 |
| --- | --- | --- |
| 结构化数据 | 商品 CSV、价格、库存、卖点 | 适合回答商品推荐、库存、价格问题 |
| 非结构化数据 | 售后政策、退换货规则、保修说明 | 适合回答规则类和长文本问题 |

整体架构：

| 模块 | 技术/文件 | 职责 |
| --- | --- | --- |
| 前端页面 | `src/app/page.tsx` | 上传文件、查看知识库、发送聊天消息、展示引用 |
| API 接口 | `src/app/api/.../route.ts` | 接收前端请求，调用 RAG 引擎 |
| RAG 引擎 | `src/lib/rag.ts` | 文档加载、切块、Embedding、存储、检索、重排、生成 |
| 向量库 | Milvus / MemoryVectorStore | 保存向量化后的知识片段 |
| Embedding | `Xenova/bge-small-zh-v1.5` | 把文本和用户问题转成向量 |
| LLM | OpenAI 兼容接口模型 | 重排序与最终回答生成 |

## 二、数据入库流程

数据入库由 `RAGEngine.addDocument(fileBuffer, fileName)` 负责，本质是一个 ETL 流程：

```text
Load 加载
  ↓
Split 切块
  ↓
Embed 向量化
  ↓
Store 存储
```

关键点：

- `.csv` 使用 `CSVLoader`，通常一行商品变成一个 `Document`。
- `.md`、`.txt` 按普通文本加载，再交给切分器处理。
- `.pdf` 可优先尝试 `WebPDFLoader`，失败后回退到基础 `PDFLoader`。
- 切块使用 `RecursiveCharacterTextSplitter`，通过 `chunkSize`、`chunkOverlap` 和中文分隔符尽量保证语义完整。
- Embedding 使用本地 `Xenova/bge-small-zh-v1.5`，减少远程依赖、调用成本和网络问题。

推荐记住这句话：

```text
入库质量决定检索质量，检索质量决定回答上限。
```

## 三、Milvus 与内存向量库

项目优先使用 Milvus，因为它适合持久化和真实项目；如果 Milvus 连接失败，则降级到 `MemoryVectorStore`，保证开发阶段还能跑通流程。

```text
尝试连接 Milvus
  ↓
成功：写入 Milvus
  ↓
失败：切到 MemoryVectorStore
```

两种模式的差异：

| 模式 | 优点 | 局限 |
| --- | --- | --- |
| Milvus | 持久化、可分页查询、适合生产 | 需要安装和启动服务 |
| MemoryVectorStore | 简单、适合开发兜底 | 进程重启丢失，不适合管理后台列数据 |

如果看到 Milvus 连接失败但上传仍成功，通常是因为系统自动使用了内存向量库兜底。

## 四、前端需要的 API

本讲新增了几个 Next.js App Router 接口：

| 接口 | 方法 | 作用 |
| --- | --- | --- |
| `/api/upload` | `POST` | 接收文件，调用 `rag.addDocument()` 入库 |
| `/api/reset` | `POST` | 调用 `rag.reset()` 清空知识库 |
| `/api/documents` | `GET` | 调用 `rag.getDocuments(page, pageSize)` 分页查看知识库 |
| `/api/chat` | `POST` | 调用 `rag.chat(query)` 完成问答 |

共同要求：

- 使用 `NextResponse.json()` 返回结果。
- 用 `try-catch` 做基本错误处理。
- 通过 `getRAGEngine()` 获取 RAG 引擎单例，避免重复初始化模型和向量库。

## 五、知识库管理能力

### 1. 上传文件

上传接口核心流程：

```text
读取 formData
  ↓
获取 file
  ↓
转 Buffer
  ↓
调用 rag.addDocument(buffer, filename)
  ↓
返回 chunks 数量或成功信息
```

前端上传文件时要用 `FormData`。

### 2. 重置知识库

`reset()` 要区分模式：

| 模式 | 处理 |
| --- | --- |
| 内存模式 | `this.vectorStore = null` |
| Milvus 模式 | 用 `MilvusClient.dropCollection()` 删除集合，再关闭连接 |

重置后必须清空 `vectorStore` 引用，避免继续使用旧的向量库实例。

### 3. 查看知识库

`getDocuments()` 在 Milvus 模式下大致流程：

```text
checkHealth
  ↓
hasCollection
  ↓
loadCollectionSync
  ↓
getCollectionStatistics 获取 row_count
  ↓
query + limit/offset 分页查询
```

展示时重点看这些字段：

| 字段 | 含义 |
| --- | --- |
| `langchain_primaryid` | LangChain 写入 Milvus 时生成的主键 |
| `langchain_text` | 原始文本内容，也是最重要的上下文字段 |
| `langchain_vector` | Embedding 后的向量 |
| `loc` / `line` | 原文位置信息 |
| `source` | 来源信息 |

前端展示知识库时，重点展示 `langchain_text`、`langchain_primaryid` 和 `loc` 即可，没必要把完整向量展示出来。

## 六、聊天核心链路

`RAGEngine.chat(query)` 是下半讲最核心的内容。

完整链路：

```text
检查知识库状态
  ↓
向量检索 Top 10
  ↓
LLM Rerank 打分
  ↓
过滤并保留 Top 3
  ↓
组装 Prompt
  ↓
LLM 生成回答
  ↓
返回 answer 和 sources
```

### 1. 初步检索

先用向量相似度召回候选文档：

```ts
const candidates = await vectorStore.similaritySearch(query, 10);
```

Top 10 的作用是先尽量找全，后面再精排。

### 2. LLM 重排序

把 Top 10 候选文档交给 LLM，让它按用户问题相关性打 0 到 10 分：

```json
[
  { "id": 0, "score": 9.5 },
  { "id": 1, "score": 6.0 }
]
```

处理策略：

- 只保留分数 `>= 6` 的文档。
- 按分数降序排序。
- 取 Top 3 作为最终上下文。
- 如果 LLM 返回格式异常，降级使用原始检索 Top 3。
- 如果过滤后没有结果，至少保留 Top 1，避免完全无法回答。

这一步体现了 RAG 中常见的“召回 + 精排”思路。

### 3. 生成回答

最终 Prompt 包含三部分：

```text
角色：睿智商城智能客服
限制：基于已知信息回答，不编造
输入：用户问题 + 检索到的上下文
```

返回结构：

```ts
return {
  answer: response,
  sources: retrievedDocs,
};
```

`sources` 是前端展示“引用来源”的依据。

## 七、前端页面功能

首页最终包含三块：

| 区域 | 功能 |
| --- | --- |
| 导入区域 | 上传文件、查看知识库 |
| 删除区域 | 重置知识库 |
| 聊天区域 | 输入问题、快捷问题、展示回答 |

按钮和接口对应关系：

| 前端动作 | 调用接口 |
| --- | --- |
| 导入文件 | `/api/upload` |
| 查看知识库 | `/api/documents` |
| 重置知识库 | `/api/reset` |
| 发送消息 | `/api/chat` |

聊天前端重点：

- 用户输入后调用 `/api/chat`。
- 把返回的 `answer` 显示到聊天框。
- 把返回的 `sources` 绑定到 AI 消息下方。
- 点击“引用来源”按钮后展开检索到的资料片段。

## 八、为什么要展示引用来源

RAG 回答展示引用来源很重要，因为它能解决两个问题：

1. 用户可以核验回答是否真的来自知识库。
2. 开发者可以排查检索、重排序和 Prompt 是否有问题。

引用来源通常展示：

- 文档内容：`pageContent` 或 `langchain_text`
- 来源文件：`metadata.source` 或 `source`
- 相关性分数：`metadata.relevanceScore`
- 原文位置：`loc` 或 `line`

## 九、常见问题与排查

| 问题 | 常见原因 | 排查方向 |
| --- | --- | --- |
| Milvus 连接失败 | Milvus 未安装或未启动 | 执行 `yarn db:up`，检查 `MILVUS_ADDRESS` |
| 上传成功但查看为空 | 实际写入了内存库或查询逻辑没加载集合 | 看终端日志，给 `getDocuments()` 加详细日志 |
| Milvus 写入 metadata 报错 | metadata 有嵌套对象或字段不符合 schema | 写入前把 metadata 处理成标量字段 |
| 聊天接口失败 | API Key 未配置、知识库为空、LLM 调用失败 | 检查 `.env.local`、日志和 `/api/chat` 返回 |
| Rerank JSON 解析失败 | LLM 返回了代码块或额外文本 | 清理 Markdown 标记，增加降级策略 |
| 回答编造 | Prompt 限制不够、检索结果不相关 | 强化“不编造”规则，检查 sources |

## 十、这一讲最重要的收获

1. RAG 项目的核心不是“问大模型”，而是先把资料变成可检索的知识库。
2. 一个可用的 RAG 应用需要完整链路：导入、存储、检索、重排、生成、引用。
3. Milvus 适合真实项目，但开发阶段需要 MemoryVectorStore 兜底。
4. `langchain_text` 是 Milvus 中最关键的展示字段，因为它对应原始文档内容。
5. Rerank 能提升检索准确度，但必须处理 LLM 输出不稳定的问题。
6. 引用来源是 RAG 应用可信度和可调试性的关键。

## 复习检查

可以用下面几个问题检验是否真正理解本讲：

1. `addDocument()` 为什么要经历 Load、Split、Embed、Store 四步？
2. 为什么本地 Embedding 可以减少网络和成本问题？
3. Milvus 连接失败时为什么还能上传成功？
4. `/api/upload`、`/api/reset`、`/api/documents`、`/api/chat` 分别负责什么？
5. `getDocuments()` 查询 Milvus 前为什么要先 `loadCollectionSync`？
6. 为什么前端展示知识库时重点看 `langchain_text`？
7. `similaritySearch(query, 10)` 和 Rerank 的关系是什么？
8. Rerank 失败时为什么要有降级策略？
9. 最终 Prompt 为什么必须强调“基于已知信息回答，不编造”？
10. RAG 回答为什么要展示引用来源？

## 最短版记忆

```text
第 12 讲 = RAG 电商客服完整实战。

上半部分：
数据准备 -> 文档加载 -> 文本切块 -> 本地 Embedding -> Milvus/内存向量库存储。

下半部分：
上传/重置/查询/聊天接口 -> 前端页面 -> 检索 Top 10 -> LLM Rerank Top 3 -> Prompt 生成回答 -> 展示引用来源。
```
