
# streamlit_rag_ollama.py
# ------------------------------------------------------------
# Local RAG with DeepSeek-R1 (Ollama) + Chroma + LangChain (2026)
# Streamlit UI (explicit "Build index" and "Ask" buttons)
# ------------------------------------------------------------
import os
import shutil
from pathlib import Path
from typing import List

import requests
import streamlit as st

# LangChain (modular packages)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Paths & defaults ---
DATA_DIR = Path("data")
DB_DIR = Path("chroma_db")
DEFAULT_LLM = "deepseek-r1:1.5b"
DEFAULT_EMBED = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

st.set_page_config(page_title="Local RAG (DeepSeek‑R1 via Ollama)", layout="wide", page_icon="🔎")
st.title("🔎 Local RAG with DeepSeek‑R1 (Ollama) — Streamlit")

# ---------------- Utilities ----------------
def ping_ollama(base_url: str) -> tuple[bool, str]:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        if r.ok:
            return True, "ONLINE"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"offline ({e})"

def load_local_documents(folder: Path) -> List:
    docs = []
    for p in folder.glob("**/*"):
        if p.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
        elif p.suffix.lower() in [".txt", ".md", ".html", ".htm"]:
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
    return docs

def build_or_load_vector_store(docs: List, embed_model: str, base_url: str):
    """
    If DB_DIR exists -> load existing Chroma.
    Else -> create from documents (persist_directory will be created).
    """
    embeddings = OllamaEmbeddings(base_url=base_url, model=embed_model)

    if DB_DIR.exists():
        # load existing index
        vs = Chroma(
            collection_name="local_rag",
            embedding_function=embeddings,
            persist_directory=str(DB_DIR),
        )
        return vs, False  # loaded
    else:
        # build new index
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
        chunks = splitter.split_documents(docs)
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="local_rag",
            persist_directory=str(DB_DIR),
        )
        # no vs.persist() in new API; persisted automatically
        return vs, True   # built

def build_rag_chain(vs: Chroma, llm_model: str, base_url: str):
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a careful, truthful assistant. "
         "Use ONLY the provided context to answer. "
         "If the answer is not in the context, say "
         "'I don't know based on the provided documents.' "
         "Cite sources as [#chunk_id]."),
        ("human",
         "Question: {question}\n\nContext:\n{context}\n\nAnswer with citations:")
    ])

    llm = ChatOllama(base_url=base_url, model=llm_model, temperature=0.2)
    parser = StrOutputParser()

    def format_docs(docs):
        lines = []
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "unknown")
            lines.append(f"[#{i}] ({src}) {d.page_content}")
        return "\n\n".join(lines)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )
    return chain

def save_uploaded_files(files: List[st.runtime.uploaded_file_manager.UploadedFile], to_dir: Path):
    to_dir.mkdir(exist_ok=True, parents=True)
    saved = []
    for f in files:
        dest = to_dir / f.name
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        saved.append(dest)
    return saved

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")

# Ollama URL (editable)
ollama_url = st.sidebar.text_input("OLLAMA_URL", value=DEFAULT_OLLAMA_URL)

ok, status = ping_ollama(ollama_url)
if ok:
    st.sidebar.success(f"Ollama: {status}")
else:
    st.sidebar.error(f"Ollama: {status}")

# Model choices
llm_choice = st.sidebar.selectbox("LLM (Ollama)", [DEFAULT_LLM, "deepseek-r1:7b"], index=0)
embed_choice = st.sidebar.selectbox("Embeddings (Ollama)", [DEFAULT_EMBED, "mxbai-embed-large"], index=0)

# Reindex button
if st.sidebar.button("Reindex (clear & rebuild)"):
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Index cleared. It will be rebuilt on the next build or query.")

# ---------------- Upload ----------------
st.subheader("📂 Documents")
DATA_DIR.mkdir(exist_ok=True)
uploads = st.file_uploader("Upload PDF/TXT/MD/HTML", type=["pdf","txt","md","html","htm"], accept_multiple_files=True)
if uploads:
    saved_paths = save_uploaded_files(uploads, DATA_DIR)
    st.success(f"Saved {len(saved_paths)} files to ./data")

# ---------------- Index controls ----------------
st.subheader("🧱 Index")
if st.button("Build index now"):
    with st.spinner("Indexing documents (this may take a while on CPU)..."):
        docs = load_local_documents(DATA_DIR)
        if not docs:
            st.warning("No documents found in ./data")
        else:
            vs, built = build_or_load_vector_store(docs, embed_choice, ollama_url)
            st.success("Index built." if built else "Existing index loaded.")

# ---------------- Q&A ----------------
st.subheader("💬 Ask a question")
colq1, colq2 = st.columns([4,1])
with colq1:
    user_query = st.text_input("Your question:", value="")
with colq2:
    ask = st.button("Ask")

if ask:
    with st.spinner("Retrieving & generating..."):
        docs = load_local_documents(DATA_DIR)
        if not docs:
            st.warning("No documents found in ./data")
        else:
            vs, built = build_or_load_vector_store(docs, embed_choice, ollama_url)
            rag = build_rag_chain(vs, llm_choice, ollama_url)
            try:
                answer = rag.invoke(user_query)
                st.markdown("### ✅ Answer")
                st.write(answer)
                if built:
                    st.info("Index was built just now.")
            except Exception as e:
                st.error(f"Generation error: {e}")
