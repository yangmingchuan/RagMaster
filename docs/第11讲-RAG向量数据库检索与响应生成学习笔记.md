# 第 11 讲学习笔记 | RAG 向量数据库、检索与响应生成

> 原文主题：RAG 实现详解（下）：向量数据库、检索前处理、检索后处理、响应生成。

## 学习目标

这一讲讲的是 RAG 的后半段。

学完需要掌握：

1. 向量数据如何存储。
2. 用户问题如何检索相关文档。
3. 检索前和检索后分别能做哪些优化。
4. 如何把检索结果交给大模型生成答案。

一句话概括：

> 第 10 讲负责“把资料变成向量”，第 11 讲负责“用向量找资料，再让大模型基于资料回答”。

## 一、整体流程

```text
文档 Chunk
  ↓
Embedding 向量化
  ↓
写入向量数据库
  ↓
用户提问
  ↓
问题向量化
  ↓
检索相关文档
  ↓
检索结果优化
  ↓
组装 Prompt
  ↓
大模型生成答案
```

核心链路：

```text
vectorStore -> retriever -> prompt -> llm -> outputParser
```

## 二、向量数据库

向量数据库用于存储文本向量，并支持相似度检索。

常见选择：

| 向量库 | 定位 | 适合场景 |
| --- | --- | --- |
| `MemoryVectorStore` | 内存向量库 | 测试、临时验证、无需持久化 |
| `Chroma` | 轻量级向量库 | 本地开发、中小规模知识库 |
| `Milvus` | 企业级向量库 | 大规模数据、高并发、生产环境 |

### 2.1 MemoryVectorStore

特点：

- 不需要额外安装服务。
- 数据存在内存中。
- 程序重启后数据丢失。
- 最适合快速跑通流程。

核心用法：

```ts
const vectorStore = await MemoryVectorStore.fromDocuments(
  chunks,
  embeddings
);

const retriever = vectorStore.asRetriever();
const docs = await retriever.invoke("LangChain 是什么？");
```

### 2.2 Chroma

特点：

- 轻量。
- 可以本地持久化。
- 适合学习和中小型应用。
- 需要启动 Chroma 服务。

核心用法：

```ts
const vectorStore = await Chroma.fromDocuments(
  chunks,
  embeddings,
  {
    collectionName: "rag-collection",
    url: "http://localhost:8000",
  }
);
```

### 2.3 Milvus

特点：

- 面向企业级场景。
- 支持大规模向量数据。
- 支持高性能检索和扩展。
- 部署与维护成本更高。

核心用法：

```ts
const vectorStore = await Milvus.fromDocuments(
  splitDocs,
  embeddings,
  {
    collectionName: "rag_collection",
    clientConfig: {
      address: "localhost:19530",
    },
  }
);

const results = await vectorStore.similaritySearchWithScore(query, 5);
```

## 三、检索器 Retriever

Retriever 是 LangChain 中的标准检索接口。

它做的事情很简单：

```text
输入：用户问题
输出：相关 Document[]
```

常见写法：

```ts
const retriever = vectorStore.asRetriever(3);
const docs = await retriever.invoke("什么是 RAG？");
```

这里的 `3` 表示返回 Top-3 相关文档。

## 四、直接向量检索

直接向量检索是最基础的方式。

流程：

```text
用户问题
  ↓
问题向量
  ↓
向量数据库相似度搜索
  ↓
返回 Top-K 文档块
```

优点：

- 简单。
- 快速。
- 容易实现。

缺点：

- 对用户提问方式敏感。
- 问题太短或太模糊时，可能召回不准。

适合：

- 知识库规模不大。
- 问题比较直接。
- 初始版本 RAG 系统。

## 五、检索前处理

检索前处理是在真正检索前，先优化用户问题。

目标：

> 提高召回率，让系统更容易找到相关文档。

### 5.1 MultiQuery

MultiQuery 会用大模型把一个问题改写成多个不同角度的问题，再分别检索。

例子：

```text
原问题：这个游戏难吗？

改写后：
1. 这个游戏对新手友好吗？
2. 这个游戏 Boss 战难度如何？
3. 这个游戏整体难度曲线怎么样？
```

适合：

- 用户问题模糊。
- 一个问题可能有多个理解角度。
- 直接检索容易漏掉资料。

代价：

- 增加 LLM 调用。
- 增加延迟和成本。

### 5.2 HyDE

HyDE，全称 Hypothetical Document Embedding，意思是假设文档嵌入。

它的流程是：

```text
用户问题
  ↓
LLM 生成一段假设答案
  ↓
用假设答案去检索真实文档
```

关键点：

- 假设答案不要求完全正确。
- 它只是用来生成更丰富的检索 query。
- 最终回答仍然必须基于真实检索结果。

适合：

- 用户问题很短。
- 文档内容较长。
- 问题表达和文档表达差异较大。

## 六、检索后处理

检索后处理是在文档已经被检索出来后，再优化这些文档。

目标：

> 提高查准率，减少无关内容进入 Prompt。

### 6.1 上下文压缩

上下文压缩会从检索到的文档里提取真正相关的句子。

常用组件：

```text
ContextualCompressionRetriever + LLMChainExtractor
```

适合：

- 检索结果太长。
- 文档里无关内容太多。
- Prompt token 成本太高。

代价：

- 需要额外调用 LLM。
- 延迟会增加。

### 6.2 Rerank 重排

Rerank 会对初步检索结果重新打分排序。

流程：

```text
向量检索 Top-10
  ↓
Reranker 重新判断相关性
  ↓
保留 Top-3
```

适合：

- 向量检索能找回相关文档，但排序不稳定。
- 需要把最相关内容排到最前。

常见工具：

- Cohere Rerank
- BGE Reranker
- 其他 Cross-Encoder 重排模型

### 6.3 父文档检索

父文档检索解决“小块检索准，但上下文不够”的问题。

思路：

```text
索引时：
小块用于向量检索
大块作为父文档保存

查询时：
先命中小块
再返回对应的大块
```

好处：

- 小块用于精准匹配。
- 大块用于提供完整上下文。

适合：

- 长文档。
- 技术文档。
- 法律、制度、章节型资料。

## 七、其他检索策略

| 检索方式 | 核心机制 | 适合场景 |
| --- | --- | --- |
| `VectorStoreRetriever` | 直接向量相似度检索 | 通用基础场景 |
| `MultiQueryRetriever` | 多角度改写问题 | 问题模糊、召回率低 |
| `ContextualCompressionRetriever` | 压缩或重排检索结果 | 内容太多、噪音太高 |
| `ParentDocumentRetriever` | 小块检索，大块返回 | 需要完整上下文 |
| `EnsembleRetriever` | 混合多个检索器 | 关键词 + 语义都重要 |
| `SelfQueryRetriever` | 将自然语言转成结构化过滤条件 | 需要按时间、类型、作者等元数据过滤 |

选择建议：

```text
起步：VectorStoreRetriever
漏召回：MultiQuery / Ensemble
无关内容多：压缩 / Rerank
上下文不足：ParentDocumentRetriever
有元数据条件：SelfQueryRetriever
```

## 八、响应生成

检索完成后，需要把检索到的上下文和用户问题一起交给大模型。

核心组件：

| 组件 | 作用 |
| --- | --- |
| `ChatOpenAI` | 调用大模型 |
| `PromptTemplate` | 定义提示词模板 |
| `StringOutputParser` | 输出普通文本 |
| `JsonOutputParser` | 输出结构化 JSON |

基础链路：

```text
context + question
  ↓
PromptTemplate
  ↓
LLM
  ↓
OutputParser
```

示例 Prompt：

```text
你是一个知识库问答助手。
请只基于以下上下文回答问题。
如果上下文中没有答案，请回答“资料中未提及”。

上下文：
{context}

问题：
{question}
```

最重要的约束：

```text
只能基于上下文回答。
资料中没有就说没有。
不要编造。
```

## 九、LCEL 与 pipe

LangChain 推荐使用 `.pipe()` 组合链路。

```ts
const chain = prompt
  .pipe(llm)
  .pipe(new StringOutputParser());
```

可以理解为：

```text
输入 -> Prompt -> LLM -> Parser -> 输出
```

当流程中有很多动态逻辑或自定义函数时，也可以使用：

```ts
RunnableSequence.from([...])
```

## 十、输出解析

最简单的输出解析是字符串：

```ts
new StringOutputParser()
```

如果前端或后续程序需要结构化数据，可以使用 JSON 输出：

```json
{
  "answer": "根据资料，RAG 是检索增强生成。",
  "sources": ["doc-1.md"],
  "confidence": "medium"
}
```

对应组件：

```ts
new JsonOutputParser()
```

## 十一、常见问题与处理方式

| 问题 | 可能原因 | 处理方式 |
| --- | --- | --- |
| 找不到相关文档 | query 太短或表达不一致 | MultiQuery、HyDE |
| 找到很多无关内容 | Top-K 太大或文档块太粗 | 降低 Top-K、上下文压缩 |
| 正确内容排在后面 | 向量相似度排序不够准 | Rerank |
| 回答缺少背景 | Chunk 太小 | 父文档检索 |
| 精确词查不准 | 向量检索不擅长精确匹配 | 混合关键词检索 |
| 模型编造答案 | Prompt 约束不足 | 明确只基于上下文回答 |

## 十二、复习检查

1. `MemoryVectorStore`、Chroma、Milvus 分别适合什么场景？
2. Retriever 的输入和输出是什么？
3. 直接向量检索有什么优缺点？
4. MultiQuery 和 HyDE 分别解决什么问题？
5. 上下文压缩和 Rerank 的区别是什么？
6. 父文档检索为什么能兼顾精准匹配和上下文完整？
7. 响应生成阶段 Prompt 中最重要的约束是什么？

## 小结

第 11 讲可以压缩成四个关键词：

```text
向量库 -> 检索器 -> 检索优化 -> 响应生成
```

完整理解：

1. 向量库负责存储和搜索语义向量。
2. Retriever 负责根据用户问题取回相关文档。
3. 检索前处理提高召回率，检索后处理提高准确率。
4. 响应生成把上下文和问题交给大模型，并用 Prompt 限制模型不要编造。
