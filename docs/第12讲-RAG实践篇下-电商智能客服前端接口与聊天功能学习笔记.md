# 第 12 讲学习笔记 | RAG 实践篇下：前端接口、聊天与引用来源

> 本篇主题：在电商智能客服 RAG 项目中，完成文档上传、知识库管理、数据展示、聊天问答、引用来源和功能测试。

## 学习目标

这一篇是在前面“数据入库”基础上继续完善 RAG 应用。

学完需要掌握：

1. 前端如何上传文档并写入向量库。
2. 后端需要哪些 API 支持知识库管理。
3. 如何展示 Milvus 中存储的文档数据。
4. 聊天接口如何完成检索、重排序和生成。
5. RAG 回答为什么要展示引用来源。
6. 测试过程中常见问题如何排查。

一句话概括：

> 这篇把 RAG 从“数据能入库”推进到“用户能上传、查看、提问，并看到带来源的回答”。

## 一、整体功能范围

本篇主要开发四块能力：

```text
文档上传
  ↓
知识库管理
  ↓
聊天问答
  ↓
引用来源展示
```

对应模块：

| 模块 | 作用 |
| --- | --- |
| 上传接口 | 接收文件，调用 RAG 引擎入库 |
| 重置接口 | 清空知识库 |
| 文档列表接口 | 查询向量库中的文档 |
| 首页 UI | 上传、重置、查看知识库、聊天 |
| 聊天接口 | 检索、重排、生成回答 |
| 引用来源 | 展示回答依据，方便核验 |

## 二、需要生成的 API 接口

项目使用 Next.js App Router。

接口文件都放在：

```text
src/app/api/.../route.ts
```

本篇新增三个知识库管理接口：

| 接口 | 方法 | 作用 |
| --- | --- | --- |
| `/api/upload` | `POST` | 上传文件并写入知识库 |
| `/api/reset` | `POST` | 清空知识库 |
| `/api/documents` | `GET` | 分页查询知识库文档 |

共同点：

- 使用 `NextResponse` 返回 JSON。
- 使用 `try-catch` 做错误处理。
- 都通过 `getRAGEngine()` 获取 RAG 引擎单例。

## 三、上传接口

路径：

```text
src/app/api/upload/route.ts
```

核心流程：

```text
读取 request.formData()
  ↓
获取 file
  ↓
校验 file 是否存在
  ↓
转成 Buffer
  ↓
调用 rag.addDocument(buffer, filename)
  ↓
返回处理结果
```

关键点：

```ts
const rag = await getRAGEngine();
await rag.addDocument(buffer, filename);
```

上传接口连接了前端文件上传和 RAG 数据入库逻辑。

## 四、重置接口

路径：

```text
src/app/api/reset/route.ts
```

核心流程：

```text
获取 RAG 引擎
  ↓
调用 rag.reset()
  ↓
返回成功结果
```

`reset()` 的实现要区分两种模式：

| 模式 | 处理方式 |
| --- | --- |
| 内存模式 | `this.vectorStore = null` |
| Milvus 模式 | 使用 `MilvusClient.dropCollection()` 删除集合 |

注意点：

- Milvus 删除集合后，要关闭连接。
- `finally` 中调用 `client.closeConnection()`。
- 重置后要把 `vectorStore` 置空，避免继续使用旧引用。

## 五、文档列表接口

路径：

```text
src/app/api/documents/route.ts
```

核心流程：

```text
读取 page 和 pageSize
  ↓
获取 RAG 引擎
  ↓
调用 rag.getDocuments(page, pageSize)
  ↓
返回 total 和 documents
```

`getDocuments()` 的 Milvus 查询流程：

```text
checkHealth
  ↓
hasCollection
  ↓
loadCollectionSync
  ↓
getCollectionStatistics 获取 row_count
  ↓
client.query 分页查询
```

分页参数：

```text
offset = (page - 1) * pageSize
limit = pageSize
```

需要注意：

- MemoryVectorStore 不适合直接列出全部文档，通常返回空列表。
- Milvus 查询前要确保集合已加载。
- LangChain 写入 Milvus 的字段名可能是适配器内部生成的。

## 六、Milvus 文档字段怎么看

查询 Milvus 后，可能看到类似结构：

```json
{
  "langchain_primaryid": "464180577318994021",
  "langchain_text": "商品ID: P006\n商品名称: 极客运动手环...",
  "langchain_vector": [0.030720721930265427],
  "loc": "{\"lines\":{\"from\":1,\"to\":6}}",
  "source": "blob",
  "line": 6
}
```

重要字段：

| 字段 | 含义 |
| --- | --- |
| `langchain_primaryid` | LangChain 写入 Milvus 时生成的主键 |
| `langchain_text` | 原始文档内容，即 `pageContent` |
| `langchain_vector` | Embedding 后的向量 |
| `loc` | 原始文本位置信息 |
| `source` | 来源文件或来源对象 |
| `line` | 行号信息 |

前端展示知识库时，重点展示：

```text
langchain_text
langchain_primaryid
loc
```

其中最重要的是 `langchain_text`，因为它是最终会作为上下文给大模型看的内容。

## 七、首页前端功能

首页主要包含三块：

| 区域 | 功能 |
| --- | --- |
| 导入区域 | 上传文件、查看知识库 |
| 删除区域 | 重置知识库 |
| 聊天区域 | 输入问题、展示回答、快捷问题 |

前端按钮和接口对应关系：

| 按钮 | 调用接口 |
| --- | --- |
| 导入文件 | `/api/upload` |
| 查看知识库 | `/api/documents` |
| 重置 | `/api/reset` |
| 发送消息 | `/api/chat` |

实现重点：

- 上传文件要使用 `FormData`。
- 查看知识库要解析 `documents` 数组。
- 重置后要清空页面状态。
- 聊天消息要区分用户消息和 AI 消息。

## 八、聊天接口

路径：

```text
src/app/api/chat/route.ts
```

接口输入：

```json
{
  "query": "推荐一款降噪耳机"
}
```

接口输出：

```json
{
  "answer": "推荐您看看极客降噪耳机 Pro...",
  "sources": []
}
```

核心流程：

```text
读取 query
  ↓
校验 query
  ↓
获取 RAG 引擎
  ↓
调用 rag.chat(query)
  ↓
返回 answer 和 sources
```

错误处理：

- `query` 为空返回 400。
- RAG 引擎报错返回 500。

## 九、RAGEngine.chat 的完整链路

`chat(query)` 是本篇的核心方法。

整体流程：

```text
知识库状态检查
  ↓
初步检索 Top 10
  ↓
LLM Rerank 重排序
  ↓
保留 Top 3
  ↓
组装 Prompt
  ↓
LLM 生成回答
  ↓
返回 answer + sources
```

## 十、阶段一：初步检索

从向量库中检索候选文档：

```ts
const candidates = await vectorStore.similaritySearch(query, 10);
```

说明：

- `10` 表示先召回 Top 10。
- 召回数量越大，覆盖率越高，但耗时也更高。
- 也可以使用 `similaritySearchWithScore` 返回相似度分数。

初步检索的目标：

```text
先尽量找全，再交给后续 Rerank 精排。
```

## 十一、阶段二：LLM Rerank

Rerank 的作用：

> 对初步检索出的候选文档重新打分，过滤不相关内容。

做法：

```text
把 Top 10 文档格式化成候选列表
  ↓
让 LLM 扮演“文档相关性评分专家”
  ↓
要求对每个文档按 0-10 打分
  ↓
只保留分数 >= 6 的文档
  ↓
按分数排序，取 Top 3
```

Prompt 输出格式要求：

```json
[
  { "id": 0, "score": 9.5 },
  { "id": 1, "score": 6.0 }
]
```

关键注意点：

- 必须要求模型只返回 JSON。
- 要处理模型可能返回 Markdown 代码块的情况。
- JSON 解析失败时要有降级策略。

降级策略：

```text
如果 Rerank 失败：
  使用原始检索结果 Top 3

如果 Rerank 后没有结果：
  使用原始 Top 1 兜底
```

这一步很重要，因为 LLM 输出格式并不总是稳定。

## 十二、阶段三：生成回答

拿到最终 Top 3 文档后，拼成上下文：

```text
context = retrievedDocs.map(doc => doc.pageContent).join("\n\n")
```

Prompt 模板包含三部分：

```text
角色：睿智商城智能客服
限制：基于已知信息回答，不编造
输入：用户问题 + 已知信息
```

示例结构：

```text
<角色>
你是一名“睿智商城”的智能客服。

<限制>
不要编造事实。如果信息中没有提到相关内容，请引导用户转人工。

<用户问题>
{question}

<已知信息>
{context}
```

返回值：

```ts
return {
  answer: response,
  sources: retrievedDocs,
};
```

`sources` 是后续展示引用来源的关键。

## 十三、前端对接聊天

前端发送消息时调用：

```text
POST /api/chat
```

需要做的事情：

```text
用户输入
  ↓
发送 query 到 chat 接口
  ↓
接收 answer 和 sources
  ↓
展示到聊天框
```

如果调用失败：

- 检查接口是否存在。
- 检查 `.env.local` 是否有 API Key。
- 检查 Milvus 是否启动。
- 查看终端日志。

## 十四、引用来源展示

RAG 回答最好展示引用来源。

原因：

- 用户可以核验答案依据。
- 方便发现检索错误。
- 能提升回答可信度。
- 便于调试 RAG 效果。

实现方式：

```text
chat 方法返回 sources
  ↓
前端在每条 AI 消息下展示“引用来源”
  ↓
点击后展开检索到的文档内容
```

引用来源可展示：

| 字段 | 用途 |
| --- | --- |
| `pageContent` / `langchain_text` | 文档内容 |
| `metadata.source` / `source` | 来源文件 |
| `metadata.relevanceScore` | Rerank 分数 |
| `loc` / `line` | 原文位置 |

## 十五、功能测试

### 15.1 数据导入测试

测试目标：

```text
上传 products.csv 后，系统能切块、Embedding、写入向量库。
```

成功日志通常包括：

```text
Processing document products.csv
Split into N chunks
Creating vector store
Embedding initialized
Vector store created/updated
```

### 15.2 查看知识库测试

测试目标：

```text
点击 View Knowledge Base 后，能看到 Milvus 中的文档内容。
```

如果返回为空：

- 检查文档是否真的写入。
- 检查 Milvus 集合是否存在。
- 检查 `getDocuments()` 是否加载集合。
- 给 `getDocuments()` 增加详细日志排查。

### 15.3 聊天测试

示例问题：

```text
推荐一款降噪耳机
```

理想日志链路：

```text
收到用户提问
  ↓
查询向量化
  ↓
初步召回多个文档
  ↓
LLM Rerank 打分
  ↓
保留高相关文档
  ↓
构建 Prompt
  ↓
发送给 LLM
  ↓
收到回复
```

如果效果正确，应该能命中商品：

```text
极客降噪耳机 Pro
价格 1299
现货
40dB 主动降噪
```

## 十六、常见问题

| 问题 | 原因 | 处理 |
| --- | --- | --- |
| Milvus 连接失败 | 数据库没启动 | 启动 Milvus，或临时使用内存模式 |
| 上传后写入失败 | metadata 不符合 Milvus 标量要求 | 清洗 metadata，确保字段是 string/number/boolean |
| 知识库查询为空 | 集合未加载或查询字段不对 | 检查 `loadCollectionSync` 和主键字段 |
| Rerank JSON 解析失败 | LLM 输出了非 JSON 内容 | 清理 Markdown 标记，增加降级逻辑 |
| 聊天接口报错 | API Key 缺失或模型调用失败 | 检查 `.env.local` |
| 回答没有依据 | sources 没返回或前端没展示 | 确认 `chat()` 返回 `sources` |

## 十七、本篇完成了什么

这一篇完成了 RAG 应用从“入库”到“可交互”的关键功能。

```text
上传文件
  ↓
查看知识库
  ↓
重置知识库
  ↓
聊天提问
  ↓
检索候选文档
  ↓
LLM Rerank
  ↓
生成回答
  ↓
展示引用来源
```

到这里，一个电商智能客服 RAG 应用已经具备基本可用形态。

## 十八、复习检查

1. `/api/upload`、`/api/reset`、`/api/documents` 分别做什么？
2. `reset()` 为什么要区分内存模式和 Milvus 模式？
3. `getDocuments()` 查询 Milvus 前为什么要先加载集合？
4. Milvus 中的 `langchain_text` 字段代表什么？
5. `chat()` 为什么先检索 Top 10，再 Rerank 取 Top 3？
6. LLM Rerank 为什么必须设计降级策略？
7. 为什么 RAG 回答要展示引用来源？
8. Milvus metadata 写入失败通常是什么原因？

## 小结

第 12 讲后半部分可以压缩成四个关键词：

```text
接口 -> 前端 -> 聊天 -> 引用
```

完整理解：

1. API 层把前端和 RAGEngine 连接起来。
2. 前端负责上传、查看、重置和聊天交互。
3. `chat()` 方法完成检索、重排序和生成。
4. 引用来源让 RAG 回答可追溯、可验证。
5. 日志和降级策略是 RAG 项目调试时非常重要的工程能力。
