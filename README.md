# PDF Chat - RAG Question Answering System

[Live Demo](URL)

Upload any PDF and ask questions using a Retrieval-Augmented Generation (RAG) pipeline. The app extracts PDF text, chunks it into searchable sections, stores semantic embeddings in ChromaDB, and uses Groq's LLM API to answer questions with source citations.

![Demo](demo.gif)

## Features

- PDF text extraction and chunking
- Semantic embeddings for similarity search
- ChromaDB vector store for document retrieval
- Groq LLM-powered question answering
- Source citations for grounded responses
- Streamlit interface for PDF upload and chat

## Tech Stack

- Python
- Streamlit
- ChromaDB
- sentence-transformers
- PyMuPDF
- Groq

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Akshitjoshi7/AI-Portfolio.git
cd AI-Portfolio/project-2-rag-pdf
```

2. Install dependencies:

```bash
pip install streamlit python-dotenv chromadb sentence-transformers pymupdf groq
```

3. Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

4. Run the app:

```bash
streamlit run app.py
```

## How It Works

1. Upload a PDF document.
2. The app extracts text using PyMuPDF.
3. Text is split into chunks for retrieval.
4. Chunks are embedded with sentence-transformers.
5. Embeddings are stored and searched in ChromaDB.
6. The most relevant chunks are sent to Groq for answer generation.
7. The answer is returned with source citations.
