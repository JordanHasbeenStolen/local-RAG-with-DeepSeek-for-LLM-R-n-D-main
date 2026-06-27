# Local RAG with DeepSeek‑R1 (Ollama) — Streamlit

> **Status**: working prototype (tested on WSL/Windows). Fully local, no external APIs.

---

## 🇬🇧 English

### What this repo does
This is a **Retrieval‑Augmented Generation (RAG)** app that:
- indexes your local documents (PDF/TXT/MD/HTML) into a **Chroma** vector store;
- computes embeddings via **Ollama** (`nomic-embed-text` or `mxbai-embed-large`);
- generates grounded answers with **DeepSeek‑R1** served by **Ollama**;
- runs a **Streamlit** UI for uploads, (re)indexing, and Q&A.

**Stack**
- **Ollama** (DeepSeek‑R1 `:1.5b` / `:7b`, embeddings),
- **LangChain** modular packages: `langchain-ollama`, `langchain-chroma`, `langchain-text-splitters`, `langchain-community`,
- **Chroma** (local vector DB),
- **Streamlit** (UI).

---

### Installation

> Python 3.10–3.12 recommended.

1) **Create and activate virtual environment (Linux/WSL)**:
```bash
python -m venv DeepSeekRagVenv
source DeepSeekRagVenv/bin/activate
```

2) **Install dependencies**:
```bash
pip install -U pip
pip install -r requirements.txt
```

3) **Install & run Ollama**:
```bash
sudo snap install ollama
ollama serve
```
if issue like  `Error: listen tcp 127.0.0.1:11434: bind: address already in use`

```bash
sudo snap stop ollama
sudo netstat -tulpn | grep 11434
ollama serve
```

4) **Pull models**:
```bash
ollama pull deepseek-r1:1.5b
ollama pull nomic-embed-text
```

5) **Run Streamlit app**:
```bash
streamlit run streamlit_rag_ollama.py --server.headless true
```

---

### Workflow
- Upload documents → Build index → Ask questions.
- Sidebar: choose LLM, embeddings, OLLAMA_URL.

### Performance tips
- Use `deepseek-r1:1.5b` + `nomic-embed-text` for CPU.
- Reduce `chunk_size` and `k` for speed.

### Troubleshooting
- **Chroma DB error**: delete `chroma_db` and rebuild.
- **Port 11434 busy**: `sudo snap stop ollama` then `ollama serve`.

---

## 🇷🇺 Русский

### Что делает проект
Приложение **RAG**:
- индексирует документы в **Chroma**;
- считает эмбеддинги через **Ollama**;
- генерирует ответы с **DeepSeek‑R1**;
- UI на **Streamlit**.

### Установка
```bash
python -m venv DeepSeekRagVenv
source DeepSeekRagVenv/bin/activate
pip install -r requirements.txt
sudo snap install ollama
ollama serve
ollama pull deepseek-r1:1.5b
ollama pull nomic-embed-text
streamlit run streamlit_rag_ollama.py --server.headless true
```

### Работа
1. Загрузите файлы.
2. Нажмите **Build index now**.
3. Задайте вопрос → **Ask**.

### Ограничения
- CPU медленный → используйте маленькие модели.
- Храните `chroma_db` в Linux-пути.

### Структура
```
.
├── data/
├── chroma_db/
├── streamlit_rag_ollama.py
├── requirements.txt
└── README.md
```
