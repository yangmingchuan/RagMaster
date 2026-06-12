const state = {
  messages: [],
  documents: [],
  totalDocs: 0,
  currentPage: 1,
  pageSize: 10,
  modalMode: "documents",
  knowledgeMeta: null,
};

const suggestions = [
  "推荐一款降噪耳机",
  "人体工学椅 V2 多少钱？",
  "七天无理由退货的条件是什么？",
  "数码产品保修期多久？",
];

const els = {
  fileInput: document.querySelector("#fileInput"),
  uploadStatus: document.querySelector("#uploadStatus"),
  viewDocsButton: document.querySelector("#viewDocsButton"),
  resetButton: document.querySelector("#resetButton"),
  seedDataButton: document.querySelector("#seedDataButton"),
  suggestions: document.querySelector("#suggestions"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chatForm"),
  chatInput: document.querySelector("#chatInput"),
  sendButton: document.querySelector("#sendButton"),
  storeStatus: document.querySelector("#storeStatus"),
  modal: document.querySelector("#documentsModal"),
  closeModalButton: document.querySelector("#closeModalButton"),
  modalTitle: document.querySelector("#modalTitle"),
  modalSubtitle: document.querySelector("#modalSubtitle"),
  documentsTable: document.querySelector("#documentsTable"),
  prevPageButton: document.querySelector("#prevPageButton"),
  nextPageButton: document.querySelector("#nextPageButton"),
  pageInfo: document.querySelector("#pageInfo"),
};

function init() {
  renderSuggestions();
  bindEvents();
  refreshHealth();
}

function bindEvents() {
  els.fileInput.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    await uploadFile(file);
    event.target.value = "";
  });

  els.viewDocsButton.addEventListener("click", () => openDocuments(1));
  els.resetButton.addEventListener("click", resetKnowledgeBase);
  els.seedDataButton.addEventListener("click", loadSeedData);
  els.closeModalButton.addEventListener("click", () => els.modal.close());
  els.prevPageButton.addEventListener("click", () => openDocuments(state.currentPage - 1));
  els.nextPageButton.addEventListener("click", () => openDocuments(state.currentPage + 1));

  els.chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await sendMessage(els.chatInput.value);
  });
}

async function uploadFile(file) {
  setUploadStatus(`正在处理 ${file.name}...`);
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");
    setUploadStatus(
      `${data.fileName}: Load ${data.originalDocuments} docs -> Split ${data.chunks} chunks -> ${data.store}，` +
      `chunkSize=${data.splitter?.chunkSize}, overlap=${data.splitter?.chunkOverlap}，Embedding=${data.embeddingStatus}`
    );
    await refreshHealth();
  } catch (error) {
    setUploadStatus(`上传失败：${error.message}`);
  }
}

async function loadSeedData() {
  els.seedDataButton.disabled = true;
  els.seedDataButton.textContent = "载入中...";
  try {
    await uploadDemoFile("/demo-data/products.csv", "products.csv");
    await uploadDemoFile("/demo-data/refund_policy.md", "refund_policy.md");
    addMessage({
      role: "assistant",
      content: "示例商品和售后政策已经载入。现在可以直接提问，例如“推荐一款降噪耳机”。",
      sources: [],
    });
  } finally {
    els.seedDataButton.disabled = false;
    els.seedDataButton.textContent = "载入示例资料";
  }
}

async function uploadDemoFile(url, filename) {
  const res = await fetch(url);
  const blob = await res.blob();
  const file = new File([blob], filename, { type: blob.type || "text/plain" });
  await uploadFile(file);
}

async function openDocuments(page = 1) {
  state.modalMode = "documents";
  const safePage = Math.max(1, page);
  const res = await fetch(`/api/documents?page=${safePage}&pageSize=${state.pageSize}`);
  const data = await res.json();

  state.documents = data.documents || [];
  state.totalDocs = data.total || 0;
  state.currentPage = safePage;
  state.knowledgeMeta = {
    splitter: data.splitter,
    embeddingStatus: data.embeddingStatus,
    store: data.store,
    ingestions: data.ingestions || [],
  };

  els.modalTitle.textContent = "Knowledge Base Chunks";
  renderDocumentsTable(state.documents);
  renderPagination();
  els.modal.showModal();
}

async function resetKnowledgeBase() {
  const confirmed = window.confirm("确定要清空当前内存知识库吗？");
  if (!confirmed) return;

  const res = await fetch("/api/reset", { method: "POST" });
  const data = await res.json();
  if (!res.ok) {
    setUploadStatus(`重置失败：${data.error || "unknown error"}`);
    return;
  }

  state.messages = [];
  renderMessages();
  setUploadStatus("知识库已重置");
  await refreshHealth();
}

async function sendMessage(rawText) {
  const query = rawText.trim();
  if (!query) return;

  addMessage({ role: "user", content: query });
  els.chatInput.value = "";
  setChatBusy(true);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Chat failed");
    addMessage({
      role: "assistant",
      content: data.answer,
      sources: data.sources || [],
    });
  } catch (error) {
    addMessage({ role: "error", content: error.message });
  } finally {
    setChatBusy(false);
  }
}

async function refreshHealth() {
  const res = await fetch("/api/health");
  const data = await res.json();
  els.storeStatus.textContent = `Memory Store · ${data.documents} chunks · ${data.embeddingStatus}`;
}

function setUploadStatus(text) {
  els.uploadStatus.textContent = text;
}

function setChatBusy(isBusy) {
  els.sendButton.disabled = isBusy;
  els.chatInput.disabled = isBusy;
  if (isBusy) {
    addTypingIndicator();
  } else {
    document.querySelector("#typingIndicator")?.remove();
  }
}

function addMessage(message) {
  state.messages.push(message);
  renderMessages();
}

function renderSuggestions() {
  els.suggestions.innerHTML = "";
  suggestions.forEach((text) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "quick-button";
    button.textContent = text;
    button.addEventListener("click", () => sendMessage(text));
    els.suggestions.appendChild(button);
  });
}

function renderMessages() {
  els.messages.innerHTML = "";

  if (state.messages.length === 0) {
    els.messages.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24"><path d="M14 4H6.5A1.5 1.5 0 0 0 5 5.5v13A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V9l-5-5Zm0 0v5h5M8 13h8M8 16h5" /></svg>
        </div>
        <p>先上传商品或售后资料，再向客服提问。</p>
        <span>也可以点击右上角载入示例资料快速体验。</span>
      </div>
    `;
    return;
  }

  state.messages.forEach((message, index) => {
    const row = document.createElement("div");
    row.className = `message-row ${message.role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.innerHTML = message.role === "user" ? userIcon() : botIcon();

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    const content = document.createElement("div");
    content.className = "bubble-content";
    content.textContent = message.content;
    bubble.appendChild(content);

    if (message.sources?.length) {
      const sourceButton = document.createElement("button");
      sourceButton.className = "source-button";
      sourceButton.type = "button";
      sourceButton.textContent = `查看 ${message.sources.length} 条引用来源`;
      sourceButton.addEventListener("click", () => openSources(message.sources, index));
      bubble.appendChild(sourceButton);
    }

    row.append(avatar, bubble);
    els.messages.appendChild(row);
  });

  els.messages.scrollTop = els.messages.scrollHeight;
}

function addTypingIndicator() {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.id = "typingIndicator";
  row.innerHTML = `
    <div class="avatar">${botIcon()}</div>
    <div class="bubble thinking">正在检索知识库...</div>
  `;
  els.messages.appendChild(row);
  els.messages.scrollTop = els.messages.scrollHeight;
}

function openSources(sources, messageIndex) {
  state.modalMode = "sources";
  els.modalTitle.textContent = `Query Sources #${messageIndex + 1}`;
  renderDocumentsTable(sources || []);
  els.modalSubtitle.textContent = `Showing top ${(sources || []).length} relevant chunks`;
  els.prevPageButton.hidden = true;
  els.nextPageButton.hidden = true;
  els.pageInfo.textContent = "Sources";
  els.modal.showModal();
}

function renderDocumentsTable(documents) {
  els.documentsTable.innerHTML = "";

  if (!documents.length) {
    els.documentsTable.innerHTML = `<div class="empty-table">No documents found.</div>`;
    return;
  }

  const table = document.createElement("table");
  table.className = "docs-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>Chunk</th>
        <th>Chunk Content</th>
        <th>Source</th>
        <th>Split Metadata</th>
      </tr>
    </thead>
  `;

  const tbody = document.createElement("tbody");
  documents.forEach((doc, index) => {
    const metadata = doc.metadata || {};
    const loc = parseLoc(doc.loc || metadata.loc);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><code>${escapeHtml(doc.langchain_primaryid || doc.id || String(index + 1))}</code></td>
      <td><pre>${escapeHtml(doc.langchain_text || doc.pageContent || "")}</pre></td>
      <td>${escapeHtml(doc.source || metadata.source || "Unknown")}</td>
      <td>
        <div class="meta-stack">
          ${loc ? `<span>Lines ${loc.lines.from}-${loc.lines.to}</span>` : ""}
          ${metadata.loader ? `<span>${escapeHtml(metadata.loader)}</span>` : ""}
          ${metadata.documentType ? `<span>${escapeHtml(metadata.documentType)}</span>` : ""}
          ${metadata.chunkIndex !== undefined ? `<span>Chunk #${escapeHtml(String(metadata.chunkIndex))}</span>` : ""}
          ${metadata.chunkChars !== undefined ? `<span>${escapeHtml(String(metadata.chunkChars))} chars</span>` : ""}
          ${metadata.chunkSize !== undefined ? `<span>chunkSize ${escapeHtml(String(metadata.chunkSize))}</span>` : ""}
          ${metadata.chunkOverlap !== undefined ? `<span>overlap ${escapeHtml(String(metadata.chunkOverlap))}</span>` : ""}
          ${doc.embeddingStatus ? `<span>Embedding ${escapeHtml(doc.embeddingStatus)}</span>` : ""}
          ${doc.relevanceScore !== undefined ? `<span>Rerank ${escapeHtml(String(doc.relevanceScore))}</span>` : ""}
          ${doc.score !== undefined ? `<span>Retrieval ${escapeHtml(String(doc.score))}</span>` : ""}
          ${Array.isArray(doc.tokenPreview) ? `<span>tokenPreview ${doc.tokenPreview.length}</span>` : ""}
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  els.documentsTable.appendChild(table);
}

function renderPagination() {
  const pageCount = Math.max(1, Math.ceil(state.totalDocs / state.pageSize));
  const splitter = state.knowledgeMeta?.splitter;
  const suffix = splitter ? ` · ${splitter.name} ${splitter.chunkSize}/${splitter.chunkOverlap}` : "";
  els.modalSubtitle.textContent = `${state.totalDocs} chunks · ${state.knowledgeMeta?.store || "MemoryStore"} · Embedding ${state.knowledgeMeta?.embeddingStatus || "unknown"}${suffix}`;
  els.pageInfo.textContent = `Page ${state.currentPage} / ${pageCount}`;
  els.prevPageButton.hidden = false;
  els.nextPageButton.hidden = false;
  els.prevPageButton.disabled = state.currentPage <= 1;
  els.nextPageButton.disabled = state.currentPage >= pageCount;
}

function parseLoc(loc) {
  if (!loc) return null;
  if (typeof loc === "object") return loc;
  try {
    return JSON.parse(loc);
  } catch {
    return null;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function botIcon() {
  return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3a7 7 0 0 0-7 7v3.2c0 2.8 2.2 5 5 5h4c2.8 0 5-2.2 5-5V10a7 7 0 0 0-7-7Zm-3.2 8.5a1.2 1.2 0 1 1 0-2.4 1.2 1.2 0 0 1 0 2.4Zm6.4 0a1.2 1.2 0 1 1 0-2.4 1.2 1.2 0 0 1 0 2.4ZM9 15h6" /></svg>`;
}

function userIcon() {
  return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm7 8a7 7 0 0 0-14 0" /></svg>`;
}

init();
