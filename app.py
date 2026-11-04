import os
import re

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Optional: load GROQ_API_KEY from a .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "qwen/qwen3.8-27b"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺", layout="wide")


# ---------- Cached resources ----------
@st.cache_resource(show_spinner="Loading vector database...")
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )


@st.cache_resource(show_spinner="Setting up the model...")
def build_chain(model_name: str, temperature: float, max_tokens: int, k: int, api_key: str):
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    db = load_vectorstore()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def clean_answer(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models may output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚙️ Settings")

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Groq API key", type="password", help="Or set GROQ_API_KEY in your environment.")
    else:
        st.success("Groq API key found in environment")

    model_name = st.text_input("Model", value=DEFAULT_MODEL)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    max_tokens = st.slider("Max tokens", 256, 4096, 1024, 128)
    k = st.slider("Documents to retrieve (k)", 1, 10, 3)
    show_sources = st.toggle("Show source documents", value=True)

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "Answers come only from the documents in your vector store. "
        "This tool is for information, not medical advice."
    )

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Header ----------
st.title("🩺 Medical Chatbot")
st.caption("Ask questions about the documents in your knowledge base.")

# Quick stats
if st.session_state.messages:
    questions = sum(1 for m in st.session_state.messages if m["role"] == "user")
else:
    questions = 0
c1, c2, c3 = st.columns(3)
c1.metric("Questions asked", questions)
c2.metric("Model", model_name.split("/")[-1])
c3.metric("Chunks per answer", k)

st.divider()


# ---------- Render history ----------
def render_sources(sources):
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, doc in enumerate(sources, start=1):
            meta = doc.metadata or {}
            origin = os.path.basename(str(meta.get("source", "unknown")))
            page = meta.get("page")
            label = f"**{i}. {origin}**" + (f" (page {int(page) + 1})" if isinstance(page, (int, float)) else "")
            st.markdown(label)
            st.caption(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            render_sources(msg["sources"])

# ---------- Chat input ----------
query = st.chat_input("Ask a question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if not api_key:
            st.error("Please provide your Groq API key in the sidebar.")
        elif not os.path.isdir(DB_FAISS_PATH):
            st.error(f"Vector store not found at '{DB_FAISS_PATH}'. Build it first.")
        else:
            try:
                chain = build_chain(model_name, temperature, max_tokens, k, api_key)
                with st.spinner("Searching documents and thinking..."):
                    response = chain.invoke({"query": query})
                answer = clean_answer(response["result"]) or "I don't know."
                sources = response.get("source_documents", [])

                st.markdown(answer)
                if show_sources and sources:
                    render_sources(sources)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")