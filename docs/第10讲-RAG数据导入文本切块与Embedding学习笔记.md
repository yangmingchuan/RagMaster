# 第 10 讲学习笔记 | RAG 数据导入、文本切块与 Embedding

> 原文主题：RAG 实现详解（上）：数据导入、文本切块和 Embedding  
> 原文链接：[小报童文章](https://xiaobot.net/post/b4ca5d11-5156-419e-8244-93f304983317)  
> 说明：网页正文需要 JavaScript 渲染，本笔记基于用户提供的文章文本整理。

## 学习目标

读完这一讲，需要掌握三件事：

1. 什么时候用 LangChain，什么时候用 LlamaIndex。
2. RAG 如何把不同格式的数据加载成统一的文档对象。
3. 文档如何切块，以及如何把文本转换成向量。

一句话概括本讲：

> RAG 的索引阶段，就是把原始数据加载进来，切成适合检索的小块，再用 Embedding 模型转换成向量。

## 一、LangChain 与 LlamaIndex 怎么选

LangChain 和 LlamaIndex 都常用于 RAG，但关注点不同。

| 框架 | 核心定位 | 更适合做什么 |
| --- | --- | --- |
| LangChain | 应用编排框架 | Prompt、Chains、Tools、Agents、RAG 流程编排 |
| LlamaIndex | 数据索引框架 | 数据接入、文档索引、检索优化、RAG 数据层 |

可以这样理解：

- LangChain 更像“应用流程编排器”。
- LlamaIndex 更像“私有数据接入和检索引擎”。

实际项目中，两者可以组合使用：

```text
LlamaIndex 负责数据索引和检索
LangChain 负责应用流程、工具调用和 Agent 编排
```

## 二、RAG 索引阶段的主流程

第 10 讲聚焦的是 RAG 的索引阶段。

```mermaid
flowchart LR
  A["原始数据"] --> B["Loader 数据导入"]
  B --> C["Document 统一格式"]
  C --> D["Text Splitter 文本切块"]
  D --> E["Embedding 向量化"]
  E --> F["写入向量数据库"]
```

核心产物有两个：

- `Document`：统一的文档对象，通常包含正文和元数据。
- `Embedding Vector`：文本对应的语义向量，用于后续相似度检索。

## 三、数据导入：把不同格式变成 Document

RAG 的第一步是读取数据。不同文件格式需要不同 Loader。

在 LangChain 中，常见 `Document` 结构可以简化理解为：

```ts
type Document = {
  pageContent: string;
  metadata: Record<string, unknown>;
};
```

其中：

- `pageContent` 是真正参与切块和向量化的文本。
- `metadata` 保存来源、页码、路径、标题等信息，方便后续引用和追溯。

### 3.1 常见 Loader 速查

| 数据类型 | 推荐 Loader | 结果特点 |
| --- | --- | --- |
| TXT | `TextLoader` | 通常一个文件生成一个 Document |
| JSON | `JSONLoader` | 可用 JSON Pointer 提取指定字段 |
| PDF | `PDFLoader` | 适合纯文本 PDF，可按页生成 Document |
| 复杂 PDF | `UnstructuredLoader` | 可识别标题、表格、图片文字等结构 |
| 浏览器级 PDF | `WebPDFLoader` | 基于 pdfjs，适合多栏或排版稍复杂的 PDF |
| CSV | `CSVLoader` | 通常每一行生成一个 Document |
| 网页 | `CheerioWebBaseLoader` | 抓取 HTML 并提取正文文本 |
| 目录 | `DirectoryLoader` | 批量递归加载目录文件 |

### 3.2 TXT 示例

```ts
import { TextLoader } from "@langchain/classic/document_loaders/fs/text";

const loader = new TextLoader("demo-data/sample.txt");
const docs = await loader.load();
```

适合：说明文档、日志、纯文本资料。

### 3.3 JSON 示例

```ts
import { JSONLoader } from "@langchain/classic/document_loaders/fs/json";

const loader = new JSONLoader("demo-data/sample.json", "/name");
const docs = await loader.load();
```

重点：

- 第二个参数是 JSON Pointer。
- `"/name"` 表示只提取 JSON 中的 `name` 字段。
- 如果 JSON 是数组，通常会为数组中的项目生成多个 Document。

### 3.4 PDF 读取怎么选

PDF 是 RAG 数据导入中最容易出问题的格式。

| 方案 | 适合场景 | 注意点 |
| --- | --- | --- |
| `PDFLoader` | 纯文本、简单单栏 PDF | 轻量，但复杂排版效果一般 |
| `WebPDFLoader` | 多栏、表格、排版稍复杂的 PDF | 依赖更重，但还原阅读顺序更好 |
| `UnstructuredLoader` | 商业级复杂 PDF、扫描件、表格、图片 | 可能需要 API Key、Docker 或 OCR 能力 |

简单选择：

```text
普通 PDF：先试 PDFLoader
多栏/表格 PDF：试 WebPDFLoader
扫描件/复杂版式：考虑 UnstructuredLoader
```

### 3.5 CSV 示例

```ts
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";

const loader = new CSVLoader("demo-data/sample.csv");
const docs = await loader.load();
```

CSV 通常会按“行”生成 Document。

例如：

```csv
id,name
1,Alice
```

可能转换为：

```text
id: 1
name: Alice
```

这种格式便于大模型理解结构化数据。

### 3.6 网页示例

```ts
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";

const loader = new CheerioWebBaseLoader("https://example.com");
const docs = await loader.load();
```

适合：技术文章、网页文档、博客内容。

注意：动态渲染网页、登录后内容、反爬页面，可能无法直接用 Cheerio 抓取。

### 3.7 目录批量加载

```ts
import { DirectoryLoader } from "@langchain/classic/document_loaders/fs/directory";
import { TextLoader } from "@langchain/classic/document_loaders/fs/text";
import { JSONLoader } from "@langchain/classic/document_loaders/fs/json";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

const loader = new DirectoryLoader("demo-data", {
  ".txt": (path) => new TextLoader(path),
  ".json": (path) => new JSONLoader(path, "/name"),
  ".pdf": (path) => new PDFLoader(path, { splitPages: false }),
});

const docs = await loader.load();
```

适合：知识库初始化、批量导入企业文档。

## 四、文本切块：把长文档切成可检索单元

加载完文档后，不能直接把超长文档全部向量化。通常要先切块。

切块的目标是：

- 块太大：噪音多，检索不准，Prompt 成本高。
- 块太小：上下文不足，语义不完整。
- 合适的块：既保留语义，又方便检索。

## 五、RecursiveCharacterTextSplitter

LangChain 中最常用的是 `RecursiveCharacterTextSplitter`。

它会按分隔符优先级递归切分，常见顺序是：

```text
段落换行 \n\n
  ↓
单换行 \n
  ↓
空格
  ↓
字符级硬切
```

这样能尽量保留段落、句子和单词的完整性。

示例：

```ts
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 100,
});

const chunks = await splitter.splitDocuments(docs);
```

推荐起步参数：

- 中文资料：`chunkSize` 可先设为 500 到 1000 字符。
- 重叠比例：`chunkOverlap` 可先设为 10% 到 20%。
- 后续根据检索效果调整。

## 六、常见切分器对比

| 切分器 | 原理 | 适合场景 | 优点 | 风险 |
| --- | --- | --- | --- | --- |
| `RecursiveCharacterTextSplitter` | 按段落、换行、空格递归切 | 通用文章、文档、说明书 | 语义保留较好 | 参数仍需调优 |
| `CharacterTextSplitter` | 按固定分隔符切 | 日志、每行一条的数据 | 简单、快 | 可能切断语义 |
| `TokenTextSplitter` | 按 Token 数切 | 严格控制上下文长度 | 对 LLM 上下文友好 | 可能破坏句子 |
| `MarkdownHeaderTextSplitter` | 按 Markdown 标题切 | Markdown 文档 | 保留层级结构 | 只适合结构化 Markdown |
| Code Splitter | 按函数、类等代码结构切 | 源代码知识库 | 保持代码块完整 | 语言支持有限 |

### Character 与 Token 示例

```ts
import { CharacterTextSplitter, TokenTextSplitter } from "@langchain/textsplitters";

const charSplitter = new CharacterTextSplitter({
  separator: "\n",
  chunkSize: 500,
  chunkOverlap: 100,
});

const tokenSplitter = new TokenTextSplitter({
  encodingName: "cl100k_base",
  chunkSize: 300,
  chunkOverlap: 50,
});
```

什么时候用 Token 切分？

- 需要严格避免超过模型上下文。
- 文档语言混杂，字符数不好估算 token 数。
- 生成阶段对 token 预算很敏感。

## 七、语义分块

固定长度切块不一定懂“话题边界”。语义分块会用 Embedding 计算句子之间的相似度，在话题变化的位置切开。

LlamaIndex 中可以使用 `SemanticSplitterNodeParser`：

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

splitter = SemanticSplitterNodeParser(
    buffer_size=3,
    breakpoint_percentile_threshold=90,
    embed_model=OpenAIEmbedding(),
)

nodes = splitter.get_nodes_from_documents(documents)
```

核心参数：

- `buffer_size`：计算相似度时参考的句子窗口。
- `breakpoint_percentile_threshold`：断点阈值，越低切得越细。
- `embed_model`：用于计算语义相似度的 Embedding 模型。

适合：

- 长文章
- 法律条款
- 技术文档
- 话题切换明显的资料

代价：

- 需要额外调用 Embedding。
- 成本和处理时间更高。

## 八、Embedding：把文本变成向量

Embedding 是把文本转换为语义向量的过程。

```text
文本：Hello RAG
  ↓
Embedding 模型
  ↓
向量：[0.012, -0.234, 0.881, ...]
```

向量的用途：

- 存入向量数据库。
- 和用户问题向量计算相似度。
- 找出最相关的文档块。

关键原则：

> 文档向量和问题向量要使用同一个 Embedding 模型，否则相似度计算没有意义。

## 九、OpenAI Embedding

LangChain 可以通过 `OpenAIEmbeddings` 调用 OpenAI 兼容接口。

```ts
import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
  openAIApiKey: apiKey,
  configuration: {
    baseURL: "https://your-openai-compatible-endpoint/v1",
  },
});

const vector = await embeddings.embedQuery("Hello RAG");
console.log(vector.length);
```

常见选择：

| 模型 | 特点 | 适合场景 |
| --- | --- | --- |
| `text-embedding-3-small` | 成本较低，通用效果好 | 大多数 RAG 项目起步 |
| `text-embedding-3-large` | 维度更高，效果更强 | 高精度检索、复杂领域 |
| `text-embedding-ada-002` | 老模型 | 新项目一般不优先选 |

## 十、开源 Embedding 模型

如果希望本地运行、降低调用成本，或者处理私有数据，可以考虑开源 Embedding 模型。

LangChain 示例：

```ts
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";

const embeddings = new HuggingFaceTransformersEmbeddings({
  modelName: "Xenova/bge-small-zh-v1.5",
});

const vector = await embeddings.embedQuery("你好，世界");
```

开源模型的特点：

- 不一定需要 API Key。
- 可本地推理，数据更可控。
- 首次运行通常需要下载模型权重。
- 效果、速度和部署成本取决于模型大小与硬件。

常见中文场景可以关注 BGE、M3E 等模型系列。

## 十一、学习路线建议

建议按这个顺序练习：

1. 用 `TextLoader` 加载一个 `.txt` 文件。
2. 用 `RecursiveCharacterTextSplitter` 把文档切成多个 Chunk。
3. 用 `OpenAIEmbeddings` 或开源模型生成向量。
4. 打印每个阶段的结果，理解 `Document -> Chunk -> Vector` 的变化。
5. 再尝试 PDF、CSV、网页和目录批量加载。

最小闭环：

```text
txt 文件
  ↓
TextLoader
  ↓
Document[]
  ↓
RecursiveCharacterTextSplitter
  ↓
Chunk Document[]
  ↓
Embedding
  ↓
Vector[]
```

## 十二、实战默认配置

如果没有特殊需求，可以先用这套配置起步：

| 环节 | 默认选择 |
| --- | --- |
| 通用框架 | LangChain |
| 数据索引增强 | LlamaIndex |
| 普通文本导入 | TextLoader |
| 批量目录导入 | DirectoryLoader |
| 普通 PDF | PDFLoader |
| 复杂 PDF | WebPDFLoader 或 UnstructuredLoader |
| 通用切块 | RecursiveCharacterTextSplitter |
| Chunk 大小 | 500 到 1000 中文字符 |
| Chunk 重叠 | 10% 到 20% |
| 通用 Embedding | text-embedding-3-small 或中文开源 Embedding |

## 十三、复习检查

学完这一讲，可以用下面几个问题自测：

1. LangChain 和 LlamaIndex 的核心区别是什么？
2. LangChain 的 `Document` 里通常包含哪两类信息？
3. 为什么 PDF 是 RAG 数据导入里的难点？
4. `RecursiveCharacterTextSplitter` 为什么比简单字符切分更稳？
5. Chunk 太大和太小分别有什么问题？
6. 为什么文档和问题必须使用同一个 Embedding 模型？
7. OpenAI Embedding 和本地开源 Embedding 各有什么取舍？

## 小结

这一讲的核心是 RAG 索引阶段的三个动作：

```text
数据导入 -> 文本切块 -> Embedding 向量化
```

可以把它理解为：

- Loader 负责把各种文件读成统一的 `Document`。
- Splitter 负责把长文档切成适合检索的 Chunk。
- Embedding 负责把 Chunk 转成可计算相似度的向量。

做好这三步，后面才能把向量写入数据库，并在用户提问时完成高质量检索。
