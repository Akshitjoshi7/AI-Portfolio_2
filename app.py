import hashlib
import os
import tempfile

import chromadb
import fitz
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_collection"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection(reset=False):
    client = get_chroma_client()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    return client.get_or_create_collection(name=COLLECTION_NAME)


def extract_pdf_text(pdf_file):
    text_by_page = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_path = temp_file.name

    try:
        document = fitz.open(temp_path)
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text().strip()
            if page_text:
                text_by_page.append((page_number, page_text))
        document.close()
    finally:
        os.remove(temp_path)

    return text_by_page


def chunk_text(text_by_page):
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for page_number, page_text in text_by_page:
        start = 0
        while start < len(page_text):
            chunk = page_text[start : start + CHUNK_SIZE].strip()
            if chunk:
                chunks.append(
                    {
                        "text": chunk,
                        "page": page_number,
                    }
                )
            start += step

    return chunks


def index_pdf(pdf_file):
    text_by_page = extract_pdf_text(pdf_file)
    chunks = chunk_text(text_by_page)

    if not chunks:
        raise ValueError("No readable text was found in this PDF.")

    model = load_embedding_model()
    collection = get_collection(reset=True)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts).tolist()
    ids = [f"chunk-{index}" for index in range(len(chunks))]
    metadatas = [
        {
            "page": chunk["page"],
            "chunk_index": index + 1,
        }
        for index, chunk in enumerate(chunks)
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


def retrieve_chunks(question):
    model = load_embedding_model()
    collection = get_collection()
    question_embedding = model.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "text": document,
            "metadata": metadata,
            "distance": distance,
        }
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]


def generate_answer(question, chunks):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

    context = "\n\n".join(
        f"Source {index} (page {chunk['metadata']['page']}):\n{chunk['text']}"
        for index, chunk in enumerate(chunks, start=1)
    )

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using only the provided PDF context. "
                    "If the answer is not in the context, say you do not know. "
                    "Cite sources using the source numbers provided."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def file_hash(pdf_file):
    return hashlib.sha256(pdf_file.getvalue()).hexdigest()


st.set_page_config(page_title="PDF Chat", page_icon="📄", layout="wide")
st.title("PDF Chat - RAG Question Answering System")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    current_file_hash = file_hash(uploaded_file)

    if st.session_state.get("pdf_hash") != current_file_hash:
        with st.spinner("Processing PDF and creating vector index..."):
            try:
                chunk_count = index_pdf(uploaded_file)
                st.session_state["pdf_hash"] = current_file_hash
                st.session_state["chunk_count"] = chunk_count
                st.session_state["pdf_ready"] = True
            except Exception as error:
                st.session_state["pdf_ready"] = False
                st.error(f"Failed to process PDF: {error}")

    if st.session_state.get("pdf_ready"):
        st.success(f"PDF indexed successfully with {st.session_state['chunk_count']} chunks.")

        question = st.text_input("Ask a question about the PDF")

        if question:
            with st.spinner("Retrieving sources and generating answer..."):
                try:
                    source_chunks = retrieve_chunks(question)
                    answer = generate_answer(question, source_chunks)

                    st.subheader("Answer")
                    st.write(answer)

                    st.subheader("Source Chunks")
                    for index, chunk in enumerate(source_chunks, start=1):
                        metadata = chunk["metadata"]
                        with st.expander(
                            f"Source {index} - Page {metadata['page']}, "
                            f"Chunk {metadata['chunk_index']}"
                        ):
                            st.write(chunk["text"])
                except Exception as error:
                    st.error(f"Failed to answer question: {error}")
else:
    st.info("Upload a PDF to start asking questions.")
