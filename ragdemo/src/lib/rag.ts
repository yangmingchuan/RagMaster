import { ChatOpenAI } from "@langchain/openai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document } from "@langchain/core/documents";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { WebPDFLoader } from "@langchain/community/document_loaders/web/pdf";
// 引入 transformers.js 的 env 配置，用于直接控制环境（如本地模型路径、是否允许远程下载）
import { env } from "@xenova/transformers";
import path from "path";
import fs from "fs";

// 1. 全局配置 transformers.js 环境
// 强制指向项目根目录下的 models 文件夹，并禁止远程下载，解决连接 HuggingFace 超时的问题
const projectRoot = process.cwd();
const modelsPath = path.resolve(projectRoot, "models");
env.localModelPath = modelsPath;
env.allowRemoteModels = false; // 禁止联网下载，强制使用本地模型
env.allowLocalModels = true;   // 允许加载本地模型

console.log("[RAG Config] 项目根目录:", projectRoot);
console.log("[RAG Config] 模型目录:", modelsPath);
console.log("[RAG Config] 检查模型文件是否存在:", fs.existsSync(path.join(modelsPath, "Xenova", "bge-small-zh-v1.5")));

// 配置信息
const MILVUS_CONFIG = {
  collectionName: "rag_collection", // 向量数据库中的集合名称（类似于关系型数据库的表名）
  clientConfig: {
    address: process.env.MILVUS_ADDRESS || "localhost:19530", // Milvus 连接地址
  },
};

const MILVUS_PRIMARY_FIELD = "langchain_primaryid";
const MILVUS_TEXT_FIELD = "langchain_text";
const MILVUS_VECTOR_FIELD = "langchain_vector";
const MILVUS_SOURCE_FIELD = "source";
const MILVUS_LOC_FIELD = "loc";
const MILVUS_TEXT_MAX_LENGTH = 8192;
const MILVUS_SOURCE_MAX_LENGTH = 1024;
const MILVUS_LOC_MAX_LENGTH = 2048;
const MILVUS_OUTPUT_FIELDS = [
  MILVUS_PRIMARY_FIELD,
  MILVUS_TEXT_FIELD,
  MILVUS_SOURCE_FIELD,
  MILVUS_LOC_FIELD,
];

type MilvusClientLike = any;
type VectorSearchResult = [Document, number];

function getCollectionNames(collections: any): string[] {
  return collections?.collection_names || collections?.data?.map((item: any) => item.name) || [];
}

function assertMilvusSuccess(response: any, action: string) {
  const status = response?.status || response;
  if (status?.error_code && status.error_code !== "Success") {
    throw new Error(`${action}失败: ${JSON.stringify(response)}`);
  }
}

function stringifyForMilvus(value: unknown, fallback: string = "") {
  if (value === undefined || value === null) return fallback;
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function truncateForMilvus(value: unknown, maxLength: number, fallback: string = "") {
  return stringifyForMilvus(value, fallback).slice(0, maxLength);
}

const DEEPSEEK_CONFIG = {
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://sg.uiuiapi.com/v1", // DeepSeek 或兼容 OpenAI 协议的 API 地址
};

const hasRealLLMKey = Boolean(DEEPSEEK_CONFIG.apiKey && DEEPSEEK_CONFIG.apiKey !== "mock-key");

// 单例模式：避免在开发环境热重载时重复加载模型，节省内存和时间
// 使用 global 防止 dev 模式下 HMR 导致实例重置
const globalForRag = global as unknown as { ragPromise: Promise<RAGEngine> | null };

let ragPromise: Promise<RAGEngine> | null = globalForRag.ragPromise || null;

import { Embeddings } from "@langchain/core/embeddings";
import { pipeline } from "@xenova/transformers";

// 2. 自定义 Embeddings 类
// 继承 LangChain 的 Embeddings 基类，自己实现 embedding 逻辑
// 目的是为了完全控制底层 transformers.js 的 pipeline 参数（如 cache_dir, quantized）
class LocalHuggingFaceEmbeddings extends Embeddings {
  private pipeline: any;
  private model: string;
  private cacheDir: string;

  constructor(fields: { model: string; cacheDir: string }) {
    super({});
    this.model = fields.model;
    this.cacheDir = fields.cacheDir;
  }

  // 初始化 pipeline，这是耗时操作，只会执行一次
  async _initPipeline() {
    if (!this.pipeline) {
      // 创建 feature-extraction 管道，用于生成文本向量
      this.pipeline = await pipeline("feature-extraction", this.model, {
        cache_dir: this.cacheDir, // 关键：指定本地缓存目录
        quantized: true,          // 开启量化，减少内存占用，提升速度（精度略有损失但通常可接受）
      });
    }
  }

  // 核心方法：将文档列表转换为向量数组
  async embedDocuments(documents: string[]): Promise<number[][]> {
    console.log(`[Embeddings] 正在向量化 ${documents.length} 个文档片段...`);
    await this._initPipeline();
    const results: number[][] = [];
    for (const doc of documents) {
      // 调用 pipeline 生成 embedding
      // pooling: "mean" 表示对所有 token 的向量求平均值作为句向量
      // normalize: true 表示归一化向量，方便计算余弦相似度
      // @ts-ignore
      const output = await this.pipeline(doc, { pooling: "mean", normalize: true });
      results.push(Array.from(output.data) as number[]);
    }
    console.log(`[Embeddings] 成功完成向量化。`);
    return results;
  }

  // 核心方法：将用户查询（Query）转换为向量
  async embedQuery(document: string): Promise<number[]> {
    console.log(`[Embeddings] 正在向量化查询: "${document.substring(0, 50)}${document.length > 50 ? '...' : ''}"`);
    await this._initPipeline();
    // @ts-ignore
    const output = await this.pipeline(document, { pooling: "mean", normalize: true });
    console.log(`[Embeddings] 查询向量生成完毕 (维度: ${output.data.length})`);
    return Array.from(output.data) as number[];
  }
}

export class RAGEngine {
  private vectorStore: MemoryVectorStore | null = null;
  private memoryDocuments: Document[] = [];
  private embeddings: LocalHuggingFaceEmbeddings;
  private llm: ChatOpenAI | null = null;
  private isMemoryStore: boolean = false;
  private milvusReady: boolean = false;

  constructor() {
    console.log("[RAG Constructor] 正在初始化 RAG 引擎...");

    // 初始化 Embeddings 模型
    // 使用本地下载好的 Xenova/bge-small-zh-v1.5 模型（中文效果好）
    console.log("[RAG Constructor] 正在初始化 Embeddings 模型...");
    try {
        this.embeddings = new LocalHuggingFaceEmbeddings({
          model: "Xenova/bge-small-zh-v1.5",
          cacheDir: modelsPath,
        });
        console.log("[RAG Constructor] Embeddings 模型初始化完成");
    } catch (error) {
        console.error("[RAG Constructor] Embeddings 初始化失败:", error);
        throw error;
    }

    if (hasRealLLMKey) {
      console.log("[RAG Constructor] 正在初始化大语言模型 (LLM)...");
      this.llm = new ChatOpenAI({
        modelName: "deepseek-v3", // 使用的模型名称
        apiKey: DEEPSEEK_CONFIG.apiKey,
        configuration: {
          baseURL: DEEPSEEK_CONFIG.baseURL,
        },
        temperature: 0.7, // 随机性控制：0.7 比较平衡，既有创造性又不会太发散
      });
    } else {
      console.warn("[RAG Warning] 未配置真实 DEEPSEEK_API_KEY。上传、切片、Embedding、向量入库可以运行；LLM Rerank 和生成会返回配置提示。");
    }
  }

  // 初始化方法：连接向量数据库
  async init() {
    if (this.vectorStore || this.milvusReady) return;

    // 先检查 Milvus 服务和集合状态。集合不存在不是错误，首次上传会创建。
    try {
        console.log("[RAG Init] 正在检查本地 Milvus 数据库...");
        await this.withMilvusClient(async (client) => {
            await client.checkHealth();
            const collections = await client.showCollections();
            const collectionNames = getCollectionNames(collections);
            if (!collectionNames.includes(MILVUS_CONFIG.collectionName)) {
                console.log("[RAG Init] Milvus 连接正常，但集合不存在，将在首次上传时创建。");
                this.isMemoryStore = false;
                return;
            }

            await this.validateMilvusCollection(client);
            await client.loadCollectionSync({ collection_name: MILVUS_CONFIG.collectionName });
            this.isMemoryStore = false;
            this.milvusReady = true;
            console.log("[RAG Init] 成功连接到现有的 Milvus 集合。");
        });
    } catch (e: any) {
        console.warn("[RAG Init] 连接 Milvus 失败，回退到 MemoryVectorStore:", e);
        this.isMemoryStore = true;
        if (this.isMemoryStore) {
            console.log("[RAG Init] 使用 MemoryVectorStore (内存版)。");
        } else {
            console.log("[RAG Init] 将使用 Milvus。");
        }
    }
  }

  private async withMilvusClient<T>(fn: (client: MilvusClientLike) => Promise<T>): Promise<T> {
    const { MilvusClient } = await import("@zilliz/milvus2-sdk-node");
    const client = new MilvusClient(MILVUS_CONFIG.clientConfig.address);
    try {
      return await fn(client);
    } finally {
      await client.closeConnection();
    }
  }

  private async hasMilvusCollection(client: MilvusClientLike) {
    const collections = await client.showCollections();
    return getCollectionNames(collections).includes(MILVUS_CONFIG.collectionName);
  }

  private async validateMilvusCollection(client: MilvusClientLike, expectedDim?: number) {
    const desc = await client.describeCollection({
      collection_name: MILVUS_CONFIG.collectionName,
    });
    assertMilvusSuccess(desc, "读取 Milvus 集合结构");

    const fields = desc?.schema?.fields || [];
    const fieldNames = new Set(fields.map((field: any) => field.name));
    const requiredFields = [
      MILVUS_PRIMARY_FIELD,
      MILVUS_TEXT_FIELD,
      MILVUS_VECTOR_FIELD,
      MILVUS_SOURCE_FIELD,
      MILVUS_LOC_FIELD,
    ];
    const missingFields = requiredFields.filter((field) => !fieldNames.has(field));
    if (missingFields.length > 0) {
      throw new Error(
        `Milvus 集合 ${MILVUS_CONFIG.collectionName} 的字段与本地 RAG schema 不一致，缺少: ${missingFields.join(", ")}。请先重置知识库后重新上传。`
      );
    }

    const vectorField =
      desc?.anns_fields?.[MILVUS_VECTOR_FIELD] ||
      fields.find((field: any) => field.name === MILVUS_VECTOR_FIELD);
    const dim =
      Number(vectorField?.dim) ||
      Number(vectorField?.type_params?.find((item: any) => item.key === "dim")?.value);
    if (expectedDim && dim && dim !== expectedDim) {
      throw new Error(
        `Milvus 集合向量维度为 ${dim}，当前 Embedding 维度为 ${expectedDim}。请重置知识库后重新上传。`
      );
    }
  }

  private async ensureMilvusCollection(client: MilvusClientLike, dimension: number) {
    const hasCollection = await this.hasMilvusCollection(client);
    if (hasCollection) {
      await this.validateMilvusCollection(client, dimension);
      await client.loadCollectionSync({ collection_name: MILVUS_CONFIG.collectionName });
      return;
    }

    const { DataType } = await import("@zilliz/milvus2-sdk-node");
    console.log(`[Milvus] 创建集合 ${MILVUS_CONFIG.collectionName}，向量维度: ${dimension}`);
    const createResult = await client.createCollection({
      collection_name: MILVUS_CONFIG.collectionName,
      fields: [
        {
          name: MILVUS_PRIMARY_FIELD,
          data_type: DataType.Int64,
          is_primary_key: true,
          autoID: true,
        },
        {
          name: MILVUS_TEXT_FIELD,
          data_type: DataType.VarChar,
          max_length: MILVUS_TEXT_MAX_LENGTH,
        },
        {
          name: MILVUS_SOURCE_FIELD,
          data_type: DataType.VarChar,
          max_length: MILVUS_SOURCE_MAX_LENGTH,
        },
        {
          name: MILVUS_LOC_FIELD,
          data_type: DataType.VarChar,
          max_length: MILVUS_LOC_MAX_LENGTH,
        },
        {
          name: MILVUS_VECTOR_FIELD,
          data_type: DataType.FloatVector,
          dim: dimension,
        },
      ],
      index_params: {
        field_name: MILVUS_VECTOR_FIELD,
        index_name: `${MILVUS_VECTOR_FIELD}_idx`,
        index_type: "FLAT",
        metric_type: "COSINE",
        params: {},
      },
      enable_dynamic_field: false,
    });
    assertMilvusSuccess(createResult, "创建 Milvus 集合");
    await client.loadCollectionSync({ collection_name: MILVUS_CONFIG.collectionName });
  }

  private async addToMilvusStore(splitDocs: Document[]) {
    if (splitDocs.length === 0) return;

    const vectors = await this.embeddings.embedDocuments(
      splitDocs.map((doc) => doc.pageContent)
    );

    await this.withMilvusClient(async (client) => {
      await client.checkHealth();
      await this.ensureMilvusCollection(client, vectors[0].length);

      const rows = splitDocs.map((doc, index) => ({
        [MILVUS_TEXT_FIELD]: truncateForMilvus(doc.pageContent, MILVUS_TEXT_MAX_LENGTH),
        [MILVUS_SOURCE_FIELD]: truncateForMilvus(
          doc.metadata?.[MILVUS_SOURCE_FIELD],
          MILVUS_SOURCE_MAX_LENGTH,
          "unknown"
        ),
        [MILVUS_LOC_FIELD]: truncateForMilvus(
          doc.metadata?.[MILVUS_LOC_FIELD],
          MILVUS_LOC_MAX_LENGTH
        ),
        [MILVUS_VECTOR_FIELD]: vectors[index],
      }));

      const insertResult = await client.insert({
        collection_name: MILVUS_CONFIG.collectionName,
        data: rows,
      });
      assertMilvusSuccess(insertResult, "写入 Milvus 向量数据");
      await client.flushSync({ collection_names: [MILVUS_CONFIG.collectionName] });
      await client.loadCollectionSync({ collection_name: MILVUS_CONFIG.collectionName });
      this.milvusReady = true;
    });
  }

  private async searchMilvusWithScore(query: string, k: number): Promise<VectorSearchResult[]> {
    const queryVector = await this.embeddings.embedQuery(query);

    return this.withMilvusClient(async (client) => {
      await client.checkHealth();
      if (!(await this.hasMilvusCollection(client))) {
        throw new Error("知识库为空，请先上传文档。");
      }

      await this.validateMilvusCollection(client, queryVector.length);
      await client.loadCollectionSync({ collection_name: MILVUS_CONFIG.collectionName });
      const searchResult = await client.search({
        collection_name: MILVUS_CONFIG.collectionName,
        data: queryVector,
        anns_field: MILVUS_VECTOR_FIELD,
        limit: k,
        output_fields: MILVUS_OUTPUT_FIELDS,
        metric_type: "COSINE",
        params: {},
      });
      assertMilvusSuccess(searchResult, "Milvus 向量检索");

      return (searchResult.results || []).map((row: any) => {
        const metadata = {
          [MILVUS_PRIMARY_FIELD]: row[MILVUS_PRIMARY_FIELD] ?? row.id,
          [MILVUS_SOURCE_FIELD]: row[MILVUS_SOURCE_FIELD] || "unknown",
          [MILVUS_LOC_FIELD]: row[MILVUS_LOC_FIELD] || "",
        };
        return [
          new Document({
            pageContent: row[MILVUS_TEXT_FIELD] || "",
            metadata,
          }),
          Number(row.score ?? 0),
        ] as VectorSearchResult;
      });
    });
  }

  // 添加文档：上传、解析、切分、向量化、存储
  async addDocument(fileBuffer: Buffer, fileName: string) {
    console.log(`[Add Document] 开始处理文件: ${fileName}`);
    let docs: Document[] = [];

    // 1. 文档加载 (Load)
    if (fileName.toLowerCase().endsWith(".pdf")) {
      // PDF 处理
      const blob = new Blob([new Uint8Array(fileBuffer)]);
      try {
        console.log("[Add Document] 尝试使用 WebPDFLoader 加载 PDF...");
        const loader = new WebPDFLoader(blob, { 
            // splitPages: false // WebPDFLoader 默认按页加载，通常效果更好
        });
        docs = await loader.load();
      } catch (e) {
        console.warn("[Add Document] WebPDFLoader 失败，回退到基础 PDFLoader:", e);
        const loader = new PDFLoader(blob, { splitPages: false });
        docs = await loader.load();
      }
    } else if (fileName.toLowerCase().endsWith(".csv")) {
      // CSV 处理
      console.log("[Add Document] 检测到 CSV 文件，使用 CSVLoader 加载...");
      // CSVLoader 需要文件路径或 Blob，这里我们创建一个临时 Blob
      const blob = new Blob([new Uint8Array(fileBuffer)]);
      const loader = new CSVLoader(blob);
      docs = await loader.load();
    } else {
      // 文本处理
      const text = fileBuffer.toString("utf-8");
      docs = [new Document({ pageContent: text, metadata: { source: fileName } })];
    }
    console.log(`[Add Document] 读取到 ${docs.length} 个原始文档对象。`);

    // 2. 文本切分 (Split)
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 800,
      chunkOverlap: 100,
      separators: ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    });

    const splitDocs = await splitter.splitDocuments(docs);
    console.log(`[Add Document] 切分为 ${splitDocs.length} 个文本块 (Chunks)。`);
    
    // 3. 元数据清洗 (Metadata Cleaning) - 对于 MemoryStore 可能不是必须的，但保留好习惯
    splitDocs.forEach((doc, index) => {
        // 确保 metadata 存在
        doc.metadata = doc.metadata || {};
        
        console.log(`[Add Document] Cleaning metadata for chunk ${index}:`, JSON.stringify(doc.metadata));

        // 1. 移除已知可能导致问题的字段
        if ('blobType' in doc.metadata) {
            delete doc.metadata.blobType;
        }

        // 2. 扁平化处理：将所有元数据值转换为 JSON 字符串
        for (const key in doc.metadata) {
            // 确保没有 undefined/null 值
            if (doc.metadata[key] === undefined || doc.metadata[key] === null) {
                 delete doc.metadata[key];
                 continue;
            }

            const value = doc.metadata[key];
            if (typeof value === 'object' && value !== null) {
                doc.metadata[key] = JSON.stringify(value);
            }
        }

        const source = doc.metadata[MILVUS_SOURCE_FIELD] === "blob"
            ? fileName
            : doc.metadata[MILVUS_SOURCE_FIELD];
        doc.metadata[MILVUS_SOURCE_FIELD] = truncateForMilvus(
            source,
            MILVUS_SOURCE_MAX_LENGTH,
            fileName
        );
        doc.metadata[MILVUS_LOC_FIELD] = truncateForMilvus(
            doc.metadata[MILVUS_LOC_FIELD],
            MILVUS_LOC_MAX_LENGTH,
            JSON.stringify({ chunk: index + 1 })
        );
        
        console.log(`[Add Document] Cleaned metadata for chunk ${index}:`, JSON.stringify(doc.metadata));
    });

    // 4. 向量化并存储 (Embed & Store)
    if (!this.vectorStore) {
        await this.createVectorStore(splitDocs);
    } else {
        await this.addToVectorStore(splitDocs);
    }

    return splitDocs.length;
  }

  // 创建向量库
  async createVectorStore(splitDocs: Document[]) {
      if (!this.isMemoryStore) {
          console.log("[Add Document] 创建新的 Milvus 集合...");
          try {
              await this.addToMilvusStore(splitDocs);
              this.memoryDocuments = [...this.memoryDocuments, ...splitDocs];
              console.log("[Add Document] Milvus 集合创建成功，文档已添加。");
          } catch (e) {
              console.error("[Add Document] 创建 Milvus 集合失败，尝试回退到 MemoryVectorStore:", e);
              this.isMemoryStore = true;
              this.milvusReady = false;
              // Fallback to memory
              this.vectorStore = await MemoryVectorStore.fromDocuments(
                  splitDocs,
                  this.embeddings
              );
              this.memoryDocuments = [...this.memoryDocuments, ...splitDocs];
              console.log("[Add Document] 回退：内存向量库创建成功。");
          }
      } else {
          console.log("[Add Document] 创建新的 MemoryVectorStore...");
          this.vectorStore = await MemoryVectorStore.fromDocuments(
              splitDocs,
              this.embeddings
          );
          this.memoryDocuments = [...this.memoryDocuments, ...splitDocs];
          console.log("[Add Document] 内存向量库创建成功，文档已添加。");
      }
  }

  // 添加到现有向量库
  async addToVectorStore(splitDocs: Document[]) {
        console.log("[Add Document] 向现有向量库添加文档...");
        if (!this.isMemoryStore) {
            await this.addToMilvusStore(splitDocs);
            this.memoryDocuments = [...this.memoryDocuments, ...splitDocs];
            console.log("[Add Document] 文档已追加到 Milvus。");
            return;
        }

        if (!this.vectorStore) {
            this.vectorStore = await MemoryVectorStore.fromDocuments(
                splitDocs,
                this.embeddings
            );
        } else {
            await this.vectorStore.addDocuments(splitDocs);
        }
        this.memoryDocuments = [...this.memoryDocuments, ...splitDocs];
        console.log("[Add Document] 文档添加完成。");
  }

  // 聊天核心逻辑
  async chat(query: string) {
    console.log(`[Chat] 收到用户提问: "${query}"`);

    // 1. 检索 (Retrieval)
    console.log("[Chat] 🔍 开始在向量库中检索...");
    
    // 扩大召回范围到 Top 10，为 Rerank 提供更多候选
    let initialResults: VectorSearchResult[] = [];
    if (this.isMemoryStore) {
        if (!this.vectorStore) {
            console.error("[Chat] 内存知识库为空。");
            throw new Error("知识库为空，请先上传文档。");
        }
        initialResults = await this.vectorStore.similaritySearchWithScore(query, 10);
    } else {
        initialResults = await this.searchMilvusWithScore(query, 10);
    }
    console.log(`[Chat] ✅ 初步召回 ${initialResults.length} 个文档片段。`);

    if (initialResults.length === 0) {
        return {
            answer: "知识库中没有检索到相关片段，请确认已经上传了包含该问题信息的文档。",
            sources: []
        };
    }

    if (!this.llm) {
        const retrievedDocs = initialResults.slice(0, 3).map(([doc, score]) => {
            // @ts-ignore
            doc.score = score;
            // @ts-ignore
            doc.relevanceScore = 0;
            return doc;
        });
        return {
            answer: "已完成真实 Embedding 向量检索，但当前没有配置真实 DEEPSEEK_API_KEY，所以未执行 LLM Rerank 和最终大模型生成。请在 .env.local 中配置 DEEPSEEK_API_KEY 后重启项目。",
            sources: retrievedDocs
        };
    }

    // 2. 重排序 (Rerank)
    console.log("[Chat] ⚖️ 开始进行 LLM Rerank (批量模式)...");
    
    // 构造带 ID 的文档列表
    const candidatesText = initialResults.map(([doc], index) => {
        return `[文档ID: ${index}]\n内容: ${doc.pageContent}`;
    }).join("\n\n------------------------\n\n");

    const rerankPrompt = `
    你是一个文档相关性评分专家。
    请判断以下文档片段与用户问题的相关性，并给出 0-10 的评分。
    
    用户问题: "${query}"
    
    候选文档列表:
    ${candidatesText}
    
    请严格按照 JSON 格式输出评分结果，不要包含任何 Markdown 标记或额外文字。
    格式示例：
    [{"id": 0, "score": 9.5}, {"id": 1, "score": 3}]
    `;

    const rerankedDocs: Document[] = [];

    try {
        const result = await this.llm.invoke(rerankPrompt);
        // 清理可能存在的 Markdown 代码块标记
        const cleanContent = String(result.content)
            .replace(/```json/g, "")
            .replace(/```/g, "")
            .trim();
        
        const scores: {id: number, score: number}[] = JSON.parse(cleanContent);
        
        console.log("[Rerank Debug] 批量打分结果:", JSON.stringify(scores));

        // 将分数回填并过滤
        scores.forEach(item => {
            if (item.id >= 0 && item.id < initialResults.length) {
                const [doc, originalScore] = initialResults[item.id];
                const relevanceScore = item.score;
                
                if (relevanceScore >= 6) {
                    // @ts-ignore
                    doc.score = originalScore;
                    // @ts-ignore
                    doc.relevanceScore = relevanceScore;
                    rerankedDocs.push(doc);
                }
            }
        });

    } catch (e) {
        console.error("[Rerank Error] 批量打分失败或解析错误，降级为使用原始 Top 3:", e);
        // 降级策略：直接使用向量检索的前 3 个
        initialResults.slice(0, 3).forEach(([doc, score]) => {
            // @ts-ignore
            doc.score = score;
            // @ts-ignore
            doc.relevanceScore = 0; // 标记为未重排序
            rerankedDocs.push(doc);
        });
    }

    // 按 LLM 打分降序排列
    // @ts-ignore
    rerankedDocs.sort((a, b) => b.relevanceScore - a.relevanceScore);
    
    // 截取 Top 3 作为最终上下文
    const finalDocs = rerankedDocs.slice(0, 3);
    console.log(`[Chat] 🎯 Rerank 完成，保留 ${finalDocs.length} 个高相关文档。`);

    // 如果 Rerank 后没有文档，使用 Top 1 作为兜底，避免完全无法回答
    const retrievedDocs = finalDocs.length > 0 ? finalDocs : initialResults.slice(0, 1).map(r => r[0]);

    // 打印最终结果用于调试
    retrievedDocs.forEach((doc, i) => {
        // @ts-ignore
        console.log(`[Chat] --- 最终片段 ${i + 1} (Rerank分: ${doc.relevanceScore}) ---`);
        console.log(doc.pageContent.substring(0, 200) + "...");
    });

    // 2. 构建提示词 (Prompt Construction)
    const template = `
    <角色>
    你是一名“睿智商城”的智能客服。请基于以下提供的商品信息或售后政策回答用户的问题。
    语气要亲切、专业。如果用户询问的商品不在已知信息中，请说明暂时缺货。
    
    <限制>
    不要编造事实。如果信息中没有提到相关内容，请礼貌地引导用户转人工服务。
    
    <用户问题>
    {question}
    
    <已知信息>
    {context}
    `;

    const prompt = PromptTemplate.fromTemplate(template);
    
    // 手动拼接上下文，用于日志打印
    const contextStr = retrievedDocs.map((d) => d.pageContent).join("\n\n");
    const formattedPrompt = await prompt.format({
        question: query,
        context: contextStr
    });
    
    console.log("\n[Chat] 🤖 构建好的完整提示词 (Prompt):");
    console.log("--------------------------------------------------");
    console.log(formattedPrompt);
    console.log("--------------------------------------------------\n");

    // 3. 生成回答 (Generation)
    console.log("[Chat] 🚀 发送请求给 LLM...");
    const chain = this.llm.pipe(new StringOutputParser());

    const response = await chain.invoke(formattedPrompt);
    console.log("[Chat] 🏁 收到 LLM 回复。");
    // 返回回答内容和来源文档，供前端展示引用
    return { answer: response, sources: retrievedDocs };
  }

  // 重置知识库
  async reset() {
    this.memoryDocuments = [];
    if (this.isMemoryStore) {
        this.vectorStore = null;
        this.milvusReady = false;
        console.log("内存向量库已重置。");
        return;
    }

    try {
      await this.withMilvusClient(async (client) => {
        const collections = await client.showCollections();
        const collectionNames = getCollectionNames(collections);
        if (collectionNames.includes(MILVUS_CONFIG.collectionName)) {
            await client.dropCollection({ collection_name: MILVUS_CONFIG.collectionName });
        }
        this.vectorStore = null; 
        this.milvusReady = false;
        console.log("Milvus 集合已删除，知识库重置成功。");
      });
    } catch (e) {
        console.error("重置数据库失败:", e);
        throw e;
    }
  }

  // 获取文档列表（用于前端表格展示）
  async getDocuments(page: number = 1, pageSize: number = 10) {
     
     if (this.isMemoryStore) {
        const offset = (page - 1) * pageSize;
        const documents = this.memoryDocuments.slice(offset, offset + pageSize).map((doc, index) => ({
            id: offset + index + 1,
            langchain_primaryid: String(offset + index + 1),
            langchain_text: doc.pageContent,
            source: doc.metadata?.source || "memory",
            loc: doc.metadata?.loc || "",
            ...doc.metadata,
            pageContent: doc.pageContent,
            metadata: doc.metadata,
        }));
        return { total: this.memoryDocuments.length, documents };
     }

     try {
         return await this.withMilvusClient(async (client) => {
         await client.checkHealth();

         const collections = await client.showCollections();
         const collectionNames = getCollectionNames(collections);
         const hasCollection = collectionNames.includes(MILVUS_CONFIG.collectionName);
         console.log(`[Get Documents] Collection '${MILVUS_CONFIG.collectionName}' exists: ${hasCollection}`);
         if (!hasCollection) return { total: 0, documents: [] };
 
         await this.validateMilvusCollection(client);

         // 确保集合已加载到内存，这是 Query 操作的前提
         await client.loadCollectionSync({ collection_name: MILVUS_CONFIG.collectionName });
 
         // 获取集合统计信息（总行数）
         const stats = await client.getCollectionStatistics({ collection_name: MILVUS_CONFIG.collectionName });
         console.log(`[Get Documents] Collection Stats:`, JSON.stringify(stats));
         const rowCountStat = stats.stats.find((s: any) => s.key === "row_count");
         const total = rowCountStat ? parseInt(String(rowCountStat.value)) : 0;
         console.log(`[Get Documents] Total rows: ${total}`);
 
         const offset = (page - 1) * pageSize;
 
         // 分页查询
         // filter: "langchain_primaryid >= 0" 是为了匹配所有记录
         // output_fields: ["*"] 表示返回所有字段（向量除外，除非显式指定向量字段名，但这里 * 通常包含标量和元数据）
         const results = await client.query({
             collection_name: MILVUS_CONFIG.collectionName,
             limit: pageSize,
             offset: offset,
             output_fields: MILVUS_OUTPUT_FIELDS, 
             filter: `${MILVUS_PRIMARY_FIELD} >= 0`,
         });
         assertMilvusSuccess(results, "查询 Milvus 文档列表");
         
         const documents = (results.data || []).map((row: any, index: number) => ({
             id: row[MILVUS_PRIMARY_FIELD] || offset + index + 1,
             ...row,
             pageContent: row[MILVUS_TEXT_FIELD],
             metadata: {
                 [MILVUS_PRIMARY_FIELD]: row[MILVUS_PRIMARY_FIELD],
                 [MILVUS_SOURCE_FIELD]: row[MILVUS_SOURCE_FIELD],
                 [MILVUS_LOC_FIELD]: row[MILVUS_LOC_FIELD],
             },
         }));
 
         return { total, documents };
         });
     } catch (e) {
         console.error("获取文档列表失败:", e);
         return { total: 0, documents: [] };
     }
  }
}

// 导出获取 RAGEngine 的单例方法
export async function getRAGEngine() {
  if (!ragPromise) {
    ragPromise = (async () => {
        const engine = new RAGEngine();
        await engine.init();
        return engine;
    })();
    globalForRag.ragPromise = ragPromise;
  }
  return ragPromise;
}
