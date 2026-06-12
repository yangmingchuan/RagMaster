import http from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 4012);

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".csv": "text/csv; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".svg": "image/svg+xml",
};

const KEY_PHRASES = [
  "降噪耳机",
  "人体工学椅",
  "智能护眼台灯",
  "充电宝",
  "机械键盘",
  "运动手环",
  "七天无理由",
  "无理由退货",
  "质量问题",
  "退换货",
  "保修",
  "质保",
  "物流",
  "发货",
  "价格保护",
  "数码",
  "家居",
  "库存",
  "现货",
  "缺货",
  "价格",
  "多少钱",
  "推荐",
];

const SPLITTER_CONFIG = {
  chunkSize: 800,
  chunkOverlap: 100,
  separators: ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
};

class LocalRAGEngine {
  constructor() {
    this.documents = [];
    this.ingestions = [];
    this.sequence = 1;
  }

  async addDocument(fileBuffer, fileName) {
    const text = fileBuffer.toString("utf8").replace(/^\uFEFF/, "");
    const rawDocuments = this.loadFileToDocuments(text, fileName);
    const chunks = await this.splitDocuments(rawDocuments, fileName);
    const ingestion = {
      fileName,
      loader: rawDocuments[0]?.metadata.loader || "unknown",
      originalDocuments: rawDocuments.length,
      chunks: chunks.length,
      splitter: {
        name: "RecursiveCharacterTextSplitter",
        ...SPLITTER_CONFIG,
      },
      store: "MemoryStore",
      embeddingStatus: "not-generated",
      createdAt: new Date().toISOString(),
    };

    this.documents.push(...chunks);
    this.ingestions.push(ingestion);
    console.log(`[RAG] ${fileName}: Load ${rawDocuments.length} docs -> Split ${chunks.length} chunks -> Store memory total=${this.documents.length}`);
    return ingestion;
  }

  reset() {
    this.documents = [];
    this.ingestions = [];
    this.sequence = 1;
    console.log("[RAG] Memory knowledge base reset.");
  }

  getDocuments(page = 1, pageSize = 10) {
    const safePage = Math.max(1, page);
    const safePageSize = Math.min(Math.max(1, pageSize), 100);
    const start = (safePage - 1) * safePageSize;
    return {
      total: this.documents.length,
      documents: this.documents.slice(start, start + safePageSize),
      ingestions: this.ingestions,
      splitter: {
        name: "RecursiveCharacterTextSplitter",
        ...SPLITTER_CONFIG,
      },
      embeddingStatus: "not-generated",
      store: "MemoryStore",
    };
  }

  getHealth() {
    return {
      ok: true,
      store: "memory",
      documents: this.documents.length,
      ingestions: this.ingestions,
      splitter: {
        name: "RecursiveCharacterTextSplitter",
        ...SPLITTER_CONFIG,
      },
      embeddingStatus: "not-generated",
    };
  }

  chat(query) {
    if (!query || !query.trim()) {
      throw new Error("Query is required");
    }

    if (this.documents.length === 0) {
      return {
        answer: "知识库现在是空的。请先上传 products.csv、refund_policy.md 或其他资料，我再基于资料回答。",
        sources: [],
      };
    }

    const initialResults = this.documents
      .map((doc) => ({ doc, score: scoreDocument(query, doc.langchain_text) }))
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);

    const fallbackResults = initialResults.length > 0
      ? initialResults
      : this.documents.slice(0, 1).map((doc) => ({ doc, score: 0.1 }));

    const topScore = Math.max(fallbackResults[0]?.score || 1, 1);
    const reranked = fallbackResults
      .map((item, index) => {
        const relativeScore = item.score > 0 ? item.score / topScore : 0.08;
        const positionBoost = index === 0 ? 0.6 : 0;
        const relevanceScore = Math.min(10, Math.max(1, 1 + relativeScore * 8.4 + positionBoost));
        return {
          ...item.doc,
          score: Number(item.score.toFixed(4)),
          relevanceScore: Number(relevanceScore.toFixed(1)),
          metadata: {
            ...item.doc.metadata,
            score: Number(item.score.toFixed(4)),
            relevanceScore: Number(relevanceScore.toFixed(1)),
          },
        };
      })
      .filter((doc, index) => index === 0 || doc.relevanceScore >= 3)
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, 3);

    const finalDocs = reranked.length > 0 ? reranked : [this.documents[0]];
    return {
      answer: generateAnswer(query, finalDocs),
      sources: finalDocs,
    };
  }

  loadFileToDocuments(text, fileName) {
    if (fileName.toLowerCase().endsWith(".csv")) {
      return this.loadCsv(text, fileName);
    }
    return this.loadText(text, fileName);
  }

  loadCsv(text, fileName) {
    const lines = text.split(/\r?\n/);
    const headerIndex = lines.findIndex((line) => line.trim());
    if (headerIndex === -1) return [];

    const headers = parseCsvLine(lines[headerIndex]);
    const docs = [];
    lines.slice(headerIndex + 1).forEach((line, lineOffset) => {
      if (!line.trim()) return;
      const lineNumber = headerIndex + lineOffset + 2;
      const values = parseCsvLine(line);
      const record = {};
      headers.forEach((header, index) => {
        record[header] = values[index] || "";
      });
      const content = headers.map((header) => `${header}: ${record[header]}`).join("\n");
      docs.push({
        pageContent: content,
        metadata: {
          source: fileName,
          loader: "CSVLoader-compatible",
          documentType: "csv-row",
          row: docs.length + 1,
          loc: { lines: { from: lineNumber, to: lineNumber } },
          originalLine: line,
        },
      });
    });
    return docs;
  }

  loadText(text, fileName) {
    const normalized = text.replace(/\r\n/g, "\n").trim();
    if (!normalized) return [];
    return [{
      pageContent: normalized,
      metadata: {
        source: fileName,
        loader: fileName.toLowerCase().endsWith(".md") ? "TextLoader(markdown)" : "TextLoader",
        documentType: fileName.toLowerCase().endsWith(".md") ? "markdown-document" : "text-document",
        loc: {
          lines: {
            from: 1,
            to: countLinesBefore(normalized, normalized.length) + 1,
          },
        },
      },
    }];
  }

  async splitDocuments(rawDocuments, fileName) {
    const splitter = await createTextSplitter();
    const splitDocs = [];

    for (let docIndex = 0; docIndex < rawDocuments.length; docIndex += 1) {
      const rawDoc = rawDocuments[docIndex];
      const pieces = await splitter.splitText(rawDoc.pageContent);
      let searchFrom = 0;

      pieces.forEach((piece, pieceIndex) => {
        const trimmedPiece = piece.trim();
        if (!trimmedPiece) return;

        let start = rawDoc.pageContent.indexOf(piece, searchFrom);
        if (start === -1) start = rawDoc.pageContent.indexOf(trimmedPiece, searchFrom);
        if (start === -1) start = searchFrom;
        const end = Math.min(rawDoc.pageContent.length, start + piece.length);
        searchFrom = Math.max(start + 1, end - SPLITTER_CONFIG.chunkOverlap);

        const baseLine = rawDoc.metadata.loc?.lines?.from || 1;
        const loc = rawDoc.metadata.documentType === "csv-row"
          ? rawDoc.metadata.loc
          : {
              lines: {
                from: baseLine + countLinesBefore(rawDoc.pageContent, start),
                to: baseLine + countLinesBefore(rawDoc.pageContent, end),
              },
            };

        splitDocs.push(this.createStoredDocument({
          content: trimmedPiece,
          source: fileName,
          loc,
          extraMetadata: {
            ...rawDoc.metadata,
            loc,
            originalDocumentIndex: docIndex,
            chunkIndex: pieceIndex,
            chunkStart: start,
            chunkEnd: end,
            chunkChars: trimmedPiece.length,
            splitter: "RecursiveCharacterTextSplitter",
            chunkSize: SPLITTER_CONFIG.chunkSize,
            chunkOverlap: SPLITTER_CONFIG.chunkOverlap,
          },
        }));
      });
    }

    return splitDocs;
  }

  createStoredDocument({ content, source, loc, extraMetadata = {} }) {
    const id = `mem_${String(this.sequence++).padStart(4, "0")}`;
    const tokenPreview = tokenize(content).slice(0, 16);
    return {
      id,
      langchain_primaryid: id,
      langchain_text: content,
      pageContent: content,
      source,
      loc: JSON.stringify(loc),
      line: loc.lines.to,
      tokenPreview,
      embeddingStatus: "not-generated",
      metadata: {
        source,
        loc,
        ...extraMetadata,
      },
    };
  }
}

const rag = new LocalRAGEngine();

const server = http.createServer(async (req, res) => {
  try {
    const requestUrl = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);

    if (requestUrl.pathname === "/api/upload" && req.method === "POST") {
      const upload = await readMultipartFile(req);
      if (!upload) {
        return sendJson(res, 400, { error: "No valid file uploaded" });
      }
      if (!/\.(csv|md|txt)$/i.test(upload.filename)) {
        return sendJson(res, 400, { error: "Only CSV, Markdown and TXT files are supported in this local demo." });
      }
      const ingestion = await rag.addDocument(upload.buffer, upload.filename);
      return sendJson(res, 200, {
        success: true,
        message: `${upload.filename} processed successfully`,
        fileName: ingestion.fileName,
        loader: ingestion.loader,
        originalDocuments: ingestion.originalDocuments,
        chunks: ingestion.chunks,
        splitter: ingestion.splitter,
        store: ingestion.store,
        embeddingStatus: ingestion.embeddingStatus,
      });
    }

    if (requestUrl.pathname === "/api/reset" && req.method === "POST") {
      rag.reset();
      return sendJson(res, 200, { success: true, message: "Knowledge base reset successfully" });
    }

    if (requestUrl.pathname === "/api/documents" && req.method === "GET") {
      const page = Number(requestUrl.searchParams.get("page") || "1");
      const pageSize = Number(requestUrl.searchParams.get("pageSize") || "10");
      return sendJson(res, 200, rag.getDocuments(page, pageSize));
    }

    if (requestUrl.pathname === "/api/chat" && req.method === "POST") {
      const body = await readJson(req);
      if (!body.query) {
        return sendJson(res, 400, { error: "Query is required" });
      }
      return sendJson(res, 200, rag.chat(String(body.query)));
    }

    if (requestUrl.pathname === "/api/health" && req.method === "GET") {
      return sendJson(res, 200, rag.getHealth());
    }

    return serveStatic(requestUrl.pathname, res);
  } catch (error) {
    console.error("[Server Error]", error);
    return sendJson(res, 500, { error: error.message || "Internal Server Error" });
  }
});

server.listen(PORT, () => {
  console.log(`RagMaster local RAG demo running at http://localhost:${PORT}`);
});

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(JSON.stringify(payload));
}

async function serveStatic(pathname, res) {
  const safePath = decodeURIComponent(pathname === "/" ? "/index.html" : pathname);
  const filePath = path.normalize(path.join(__dirname, safePath));

  if (!filePath.startsWith(__dirname)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  try {
    const file = await readFile(filePath);
    const ext = path.extname(filePath);
    res.writeHead(200, {
      "Content-Type": MIME_TYPES[ext] || "application/octet-stream",
      "Cache-Control": "no-store",
    });
    res.end(file);
  } catch {
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Not found");
  }
}

function readRequestBuffer(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

async function readJson(req) {
  const buffer = await readRequestBuffer(req);
  if (!buffer.length) return {};
  return JSON.parse(buffer.toString("utf8"));
}

async function readMultipartFile(req) {
  const contentType = req.headers["content-type"] || "";
  const boundaryMatch = contentType.match(/boundary=(?:"([^"]+)"|([^;]+))/i);
  if (!boundaryMatch) return null;

  const boundary = boundaryMatch[1] || boundaryMatch[2];
  const body = (await readRequestBuffer(req)).toString("latin1");
  const parts = body.split(`--${boundary}`);

  for (const part of parts) {
    if (!part.includes('name="file"')) continue;
    const headerEnd = part.indexOf("\r\n\r\n");
    if (headerEnd === -1) continue;

    const rawHeaders = part.slice(0, headerEnd);
    const filename = decodeMultipartFilename(rawHeaders);
    let content = part.slice(headerEnd + 4);
    content = content.replace(/\r\n$/, "");
    return {
      filename: path.basename(filename),
      buffer: Buffer.from(content, "latin1"),
    };
  }

  return null;
}

function decodeMultipartFilename(rawHeaders) {
  const encodedMatch = rawHeaders.match(/filename\*=UTF-8''([^;\r\n]+)/i);
  if (encodedMatch) {
    return decodeURIComponent(encodedMatch[1]);
  }

  const filenameMatch = rawHeaders.match(/filename="([^"]*)"/i);
  const rawFilename = filenameMatch?.[1] || "uploaded.txt";
  return Buffer.from(rawFilename, "latin1").toString("utf8");
}

async function createTextSplitter() {
  try {
    const { RecursiveCharacterTextSplitter } = await import("@langchain/textsplitters");
    return new RecursiveCharacterTextSplitter(SPLITTER_CONFIG);
  } catch (error) {
    console.warn("[RAG] @langchain/textsplitters not available, using local RecursiveCharacterTextSplitter-compatible fallback.", error.message);
    return {
      splitText(text) {
        return splitTextRecursively(text, SPLITTER_CONFIG);
      },
    };
  }
}

function parseCsvLine(line) {
  const cells = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const next = line[i + 1];
    if (char === '"' && inQuotes && next === '"') {
      current += '"';
      i += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      cells.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }

  cells.push(current.trim());
  return cells;
}

function splitTextRecursively(text, config) {
  const { chunkSize, chunkOverlap, separators } = config;
  const chunks = [];
  splitBySeparator(text, separators, chunkSize).forEach((piece) => {
    if (!piece.trim()) return;
    chunks.push(piece);
  });

  const withOverlap = [];
  for (let i = 0; i < chunks.length; i += 1) {
    const previous = withOverlap[withOverlap.length - 1] || "";
    const overlap = i > 0 ? previous.slice(Math.max(0, previous.length - chunkOverlap)) : "";
    const next = `${overlap}${chunks[i]}`.trim();
    if (next) withOverlap.push(next);
  }
  return withOverlap;
}

function splitBySeparator(text, separators, chunkSize) {
  if (text.length <= chunkSize) return [text];
  if (separators.length === 0) {
    const chunks = [];
    for (let start = 0; start < text.length; start += chunkSize) {
      chunks.push(text.slice(start, start + chunkSize));
    }
    return chunks;
  }

  const [separator, ...rest] = separators;
  const splits = separator ? text.split(separator).map((part, index, arr) => index < arr.length - 1 ? `${part}${separator}` : part) : [...text];
  const chunks = [];
  let current = "";

  for (const split of splits) {
    if ((current + split).length <= chunkSize) {
      current += split;
      continue;
    }
    if (current.trim()) chunks.push(current);
    if (split.length > chunkSize) {
      chunks.push(...splitBySeparator(split, rest, chunkSize));
      current = "";
    } else {
      current = split;
    }
  }

  if (current.trim()) chunks.push(current);
  return chunks;
}

function countLinesBefore(text, offset) {
  return (text.slice(0, offset).match(/\n/g) || []).length;
}

function normalize(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[，。！？、：；（）【】《》"'“”‘’]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenize(text) {
  const normalized = normalize(text);
  const tokens = new Set();

  for (const phrase of KEY_PHRASES) {
    if (normalized.includes(phrase.toLowerCase())) {
      tokens.add(phrase.toLowerCase());
    }
  }

  const words = normalized.match(/[a-z0-9]+/g) || [];
  words.forEach((word) => tokens.add(word));

  const cjk = [...normalized].filter((char) => /\p{Script=Han}/u.test(char));
  for (let i = 0; i < cjk.length - 1; i += 1) {
    tokens.add(cjk[i] + cjk[i + 1]);
  }

  return [...tokens].filter((token) => token.length > 1);
}

function scoreDocument(query, content) {
  const terms = tokenize(query);
  const normalizedContent = normalize(content);
  let score = 0;

  for (const term of terms) {
    if (!normalizedContent.includes(term)) continue;
    const phraseBoost = KEY_PHRASES.map((item) => item.toLowerCase()).includes(term) ? 3.2 : 1;
    score += phraseBoost + Math.min(term.length / 5, 1.4);
  }

  if (normalizedContent.includes(normalize(query))) score += 5;
  if (/推荐|买哪|哪款/.test(query) && /商品名称|核心卖点|库存状态/.test(content)) score += 1.8;
  if (/价格|多少钱|售价/.test(query) && /价格:/.test(content)) score += 2;
  if (/库存|有货|现货|缺货/.test(query) && /库存状态:/.test(content)) score += 2;
  if (/退货|退换|售后|保修|质保|物流|发货|价格保护/.test(query) && /政策|退货|保修|发货|价格保护/.test(content)) score += 2.5;

  return score;
}

function generateAnswer(query, docs) {
  const top = docs[0];
  const product = parseProduct(top.langchain_text);
  const normalizedQuery = normalize(query);

  if (product) {
    const lines = [];
    if (/推荐|降噪|耳机|哪款|买/.test(query)) {
      lines.push(`推荐您优先看看「${product["商品名称"]}」。`);
    } else {
      lines.push(`我在知识库里找到了「${product["商品名称"]}」。`);
    }

    if (product["价格"]) lines.push(`价格是 ${product["价格"]} 元。`);
    if (product["库存状态"]) lines.push(`当前库存状态为：${product["库存状态"]}。`);
    if (product["核心卖点"]) lines.push(`核心卖点是：${product["核心卖点"]}。`);

    if (normalizedQuery.includes("保修") || normalizedQuery.includes("质保")) {
      const policy = docs.find((doc) => /保修|质保/.test(doc.langchain_text));
      if (policy) {
        lines.push(`保修信息可参考知识库中的售后政策：${compactText(policy.langchain_text, 90)}`);
      }
    }

    lines.push("以上回答基于当前上传的知识库片段，您可以展开引用来源核对原文。");
    return lines.join("\n");
  }

  const context = docs.map((doc) => compactText(doc.langchain_text, 160)).join("\n\n");
  if (/退货|退换|售后|保修|质保|物流|发货|价格保护/.test(query)) {
    return `根据睿智商城知识库：\n${context}\n\n如果您的具体订单情况比较特殊，建议再联系人工客服确认。`;
  }

  return `我从知识库中找到了以下相关信息：\n${context}\n\n这个本地 Demo 会严格基于已上传资料回答；如果资料里没有覆盖，建议补充文档后再问。`;
}

function parseProduct(text) {
  if (!/商品名称:/.test(text)) return null;
  const record = {};
  text.split(/\r?\n/).forEach((line) => {
    const index = line.indexOf(":");
    if (index === -1) return;
    record[line.slice(0, index).trim()] = line.slice(index + 1).trim();
  });
  return record;
}

function compactText(text, maxLength) {
  const compact = text.replace(/\s+/g, " ").trim();
  return compact.length > maxLength ? `${compact.slice(0, maxLength)}...` : compact;
}
