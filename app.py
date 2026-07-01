import hashlib
import os
from typing import List

import chromadb
import fitz
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "pdf_collection"
CHROMA_PATH = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3


load_dotenv()


@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PATH)


def extract_pdf_text(pdf_bytes: bytes) -> str:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in document]
    document.close()
    return "\n".join(pages)


def chunk_text(text: str) -> List[str]:
    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        return []

    chunks = []
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP

    while start < len(cleaned_text):
        chunk = cleaned_text[start : start + CHUNK_SIZE]
        chunks.append(chunk)
        start += step

    return chunks


def reset_collection(client: chromadb.PersistentClient):
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.get_or_create_collection(name=COLLECTION_NAME)


def index_pdf(pdf_bytes: bytes, file_name: str) -> int:
    text = extract_pdf_text(pdf_bytes)
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No readable text was found in this PDF.")

    model = load_embedding_model()
    embeddings = model.encode(chunks).tolist()

    client = get_chroma_client()
    collection = reset_collection(client)
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:12]

    collection.add(
        ids=[f"{pdf_hash}-{index}" for index in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"source": file_name, "chunk": index + 1}
            for index in range(len(chunks))
        ],
    )

    return len(chunks)


def retrieve_chunks(question: str):
    model = load_embedding_model()
    question_embedding = model.encode([question]).tolist()[0]

    client = get_chroma_client()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    return collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )


def build_prompt(question: str, chunks: List[str]) -> str:
    context = "\n\n".join(
        f"Source chunk {index + 1}:\n{chunk}" for index, chunk in enumerate(chunks)
    )
    return f"""Answer the question using only the context below.
If the answer is not in the context, say you do not know.

Context:
{context}

Question:
{question}

Answer:"""


def ask_groq(question: str, chunks: List[str]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY was not found. Add it to your .env file.")

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a precise PDF question-answering assistant.",
            },
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content


def main():
    st.set_page_config(page_title="PDF Question Answering", page_icon="📄")
    st.title("PDF Question Answering")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

        if st.session_state.get("indexed_pdf_hash") != pdf_hash:
            with st.spinner("Processing PDF and building vector index..."):
                try:
                    chunk_count = index_pdf(pdf_bytes, uploaded_file.name)
                except Exception as exc:
                    st.error(str(exc))
                    st.stop()

            st.session_state["indexed_pdf_hash"] = pdf_hash
            st.session_state["indexed_file_name"] = uploaded_file.name
            st.session_state["chunk_count"] = chunk_count

        st.success(
            f"Indexed {st.session_state['chunk_count']} chunks from "
            f"{st.session_state['indexed_file_name']}."
        )

        question = st.text_input("Ask a question about the PDF")

        if st.button("Ask", type="primary", disabled=not question.strip()):
            with st.spinner("Retrieving sources and generating answer..."):
                try:
                    results = retrieve_chunks(question)
                    source_chunks = results["documents"][0]
                    source_metadata = results["metadatas"][0]
                    answer = ask_groq(question, source_chunks)
                except Exception as exc:
                    st.error(str(exc))
                    st.stop()

            st.subheader("Answer")
            st.write(answer)

            with st.expander("Source chunks used"):
                for index, chunk in enumerate(source_chunks):
                    metadata = source_metadata[index]
                    st.markdown(
                        f"**Chunk {metadata.get('chunk', index + 1)} "
                        f"from {metadata.get('source', 'uploaded PDF')}**"
                    )
                    st.write(chunk)
    else:
        st.info("Upload a PDF to begin.")


if __name__ == "__main__":
    main()
