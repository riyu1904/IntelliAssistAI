import os
import requests
import numpy as np
import faiss
import streamlit as st

st.set_page_config(page_title="IntelliAssist AI", page_icon="📚")
from sentence_transformers import SentenceTransformer

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Please configure the GEMINI_API_KEY environment variable.")
    st.stop()

@st.cache_resource
def load_resources():
    docs_folder = os.path.join(os.path.dirname(__file__), "documents")
    documents = {}

    for filename in os.listdir(docs_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(docs_folder, filename),
                      "r", encoding="utf-8") as file:
                documents[filename] = file.read().strip()

    chunks = []
    for filename, text in documents.items():
        start = 0
        while start < len(text):
            end = start + 800
            chunks.append({"source": filename, "text": text[start:end]})
            if end >= len(text):
                break
            start = end - 150

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return documents, chunks, model, index

documents, chunks, model, index = load_resources()

def answer_question(question):
    # 1. Search your document index
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, k=3)

    retrieved_chunks = [
        chunks[i] for i in indices[0] if i >= 0
    ]

    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    # 2. Ask Gemini using the working direct API
    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/gemini-3.5-flash:generateContent"
    )

    payload = {
        "contents": [{
            "parts": [{
                "text": (
                    "You are IntelliAssist AI, a helpful document assistant.\n"
                    "Answer using only the context below. If the answer "
                    "is absent, say you could not find it in the documents.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\nAnswer:"
                )
            }]
        }],
        "generationConfig": {
            "maxOutputTokens": 512,
            "thinkingConfig": {"thinkingLevel": "minimal"}
        }
    }

    response = requests.post(
        url,
        params={"key": API_KEY},
        json=payload,
        timeout=30
    )
    response.raise_for_status()

    data = response.json()
    candidates = data.get("candidates", [])
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []

    answer = "\n".join(
        part["text"] for part in parts if "text" in part
    )

    if not answer:
        answer = "Gemini returned no text. Please try again."

    # 3. Show the sources used
    sources = list(dict.fromkeys(
        chunk["source"] for chunk in retrieved_chunks
    ))

    return answer + "\n\nSources:\n" + "\n".join(
        f"- {source}" for source in sources
    )


def summarize_document(filename):
    if filename not in documents:
        return f"Document not found: {filename}"

    text = documents[filename]

    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/gemini-3.5-flash:generateContent"
    )

    payload = {
        "contents": [{
            "parts": [{
                "text": (
                    "Summarize the following document clearly. "
                    "Provide the main idea, key points, and conclusion. "
                    "Use only information in the document.\n\n"
                    f"Document:\n{text}"
                )
            }]
        }],
        "generationConfig": {
            "maxOutputTokens": 512,
            "thinkingConfig": {"thinkingLevel": "minimal"}
        }
    }

    response = requests.post(
        url,
        params={"key": API_KEY},
        json=payload,
        timeout=30
    )
    response.raise_for_status()

    data = response.json()
    candidates = data.get("candidates", [])

    if not candidates:
        return "Gemini returned no summary."

    parts = candidates[0].get("content", {}).get("parts", [])
    summary = "\n".join(
        part["text"] for part in parts if "text" in part
    )

    return summary or "Gemini returned no summary text."


st.title("📚 IntelliAssist AI")
st.caption("Ask questions about your uploaded documents.")

mode = st.radio("Choose a feature", ["Ask a question", "Summarize a document"])

if mode == "Ask a question":
    question = st.text_input("Enter your question")
    if st.button("Get answer") and question.strip():
        with st.spinner("Searching documents and generating an answer..."):
            try:
                st.write(answer_question(question))
            except Exception as error:
                st.error(f"Could not generate answer: {error}")

else:
    filename = st.selectbox("Select a document", list(documents.keys()))
    if st.button("Summarize"):
        with st.spinner("Summarizing document..."):
            try:
                st.write(summarize_document(filename))
            except Exception as error:
                st.error(f"Could not summarize document: {error}")
