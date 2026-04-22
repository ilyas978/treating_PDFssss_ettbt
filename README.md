# 📚 Multi-PDF RAG Chatbot | Gemini 2.0 Flash & LangChain

This project is a **Retrieval-Augmented Generation (RAG)** application designed to interact with multiple PDF documents. By leveraging the latest **Gemini 2.0 Flash** model, it provides high-speed, context-aware answers based on private data provided by the user.

---

## 🌟 Key Features

* **Multi-Document Ingestion**: Upload multiple PDF files and query them simultaneously.
* **Semantic Retrieval**: Uses `FAISS` vector database to find the most relevant context, going beyond simple keyword matching.
* **Gemini 2.0 Flash Integration**: Optimized for fast and accurate natural language generation.
* **Smart Context Management**: Implements text chunking with overlap to preserve semantic meaning across document segments.
* **Safety & Reliability**: Custom prompt engineering to prevent hallucinations and ensure responses are strictly based on the provided PDFs.

---

## 🏗️ Architecture

The system follows the standard RAG lifecycle:
1.  **Extraction**: Text is extracted from PDFs using `PyPDF2`.
2.  **Chunking**: Text is split into 1000-character segments with a 200-character overlap.
3.  **Embedding**: Text chunks are converted into vectors using `GoogleGenerativeAIEmbeddings`.
4.  **Vector Store**: Embeddings are stored locally in a `FAISS` index.
5.  **Querying**: When a question is asked, the system retrieves relevant chunks and sends them to the LLM as context.



---

## 🛠️ Tech Stack

* **Language**: Python 3.9+
* **LLM**: Google Gemini 2.0 Flash
* **Orchestration**: LangChain (Core, Community, Google GenAI)
* **Vector DB**: FAISS (Facebook AI Similarity Search)
* **UI**: Streamlit
* **Environment**: Python-dotenv (Secrets Management)

---

## 🚀 Getting Started

### 1. Prerequisites
* A Google AI Studio API Key (Get it [here](https://aistudio.google.com/))
* Python installed on your machine

### 2. Installation
```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Install dependencies
pip install -r requirements.txt
