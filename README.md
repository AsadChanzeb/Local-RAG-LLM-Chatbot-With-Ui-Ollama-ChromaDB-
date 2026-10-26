# 🤖 Local RAG LLM Chatbot with UI (Ollama + ChromaDB)

A fast, private, and fully local **Retrieval-Augmented Generation (RAG)** chatbot powered by **Ollama**, **ChromaDB**, and **LangChain**, paired with a responsive modern Web UI.

Ask questions about your uploaded PDF documents with full privacy, real-time token streaming, and clickable source context inspection.

---

## 📸 User Interface Showcase

> _Drop your screenshots in the placeholders below or in an `assets/` or `images/` directory._

### 1. Main Dashboard & Model Selector

<img width="2737" height="1722" alt="ai3" src="https://github.com/user-attachments/assets/cebb226d-a71c-4fbf-ba3a-7ed0590cb27a" />


<br>

### 2. Real-Time Streaming Chat & Citations
<img width="3255" height="1720" alt="ai2" src="https://github.com/user-attachments/assets/96380ede-dac1-446a-8046-353f622d6597" />


<br>

### 3. Interactive Source Attribution & Match Confidence
<img width="2705" height="1735" alt="image" src="https://github.com/user-attachments/assets/bec09f34-b7f3-48f2-b9e8-08cf46a150a2" />


---

## ✨ Key Features

- **🔒 100% Local & Private Option**: Embed documents and generate responses on your local GPU/CPU using Ollama — zero cloud data transfer.
- **⚡ Real-Time Token Streaming**: Fast Server-Sent Events (SSE) stream responses instantly to the browser as tokens are generated.
- **📚 Smart Document Processing**:
  - Automatically loads and splits PDFs with customizable chunk size (1,000 chars) and overlap (150 chars).
  - Incremental indexing — newly uploaded PDFs are indexed without re-processing existing files.
- **🔍 Vector Retrieval with ChromaDB**: Embeds queries via `nomic-embed-text` and retrieves top-k semantically relevant chunks with distance scores.
- **🎯 Precise Source Attribution**: View the exact document name, page number, and content snippet backing each answer.
- **🔄 Multi-Model Switching**:
  - **Local (Ollama)**: Llama 3.2 (3B), Mistral (7B), or any custom Ollama model.
  - **Cloud Fallback (Optional)**: Google Gemini (2.0 Flash / Pro) and xAI Grok (3 / Mini).
- **💡 One-Click Startup**: Windows launcher batch file (`run.bat`) automatically selects the correct Python virtual environment.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend** | Python 3.11, Flask | Lightweight HTTP & SSE streaming API |
| **Orchestration** | LangChain / LangChain-Ollama | RAG prompt pipelines & model integration |
| **Vector DB** | ChromaDB | Persistent local vector embeddings storage |
| **Embeddings** | `nomic-embed-text` | Fast local embeddings via Ollama |
| **Local LLM** | Ollama (`llama3.2:3b`, `mistral`) | Local inference |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript | Modern, glassmorphic UI with zero npm dependencies |

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python**: 3.10 or 3.11 recommended
- **Ollama**: Download and install from [ollama.com](https://ollama.com)

Pull the recommended embedding and chat models:
```bash
# High-quality embedding model
ollama pull nomic-embed-text

# Fast local language model (3.2B parameters)
ollama pull llama3.2:3b

# (Optional) Accurate reasoning model (7B parameters)
ollama pull mistral
```

---

### 2. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AsadChanzeb/Local-RAG-LLM-Chatbot-With-Ui-Ollama-ChromaDB-.git
   cd Local-RAG-LLM-Chatbot-With-Ui-Ollama-ChromaDB-
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Configure environment variables:**
   If you plan to use Google Gemini or xAI Grok in addition to Ollama, copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Add your API keys inside `.env`.

---

### 3. Ingest Documents into Database

Place your PDF files inside the `data/` directory, then populate ChromaDB:
```bash
python populate_database.py
```
*(To clear and rebuild the database from scratch, run `python populate_database.py --reset`)*

---

### 4. Run the Web Application

**Option A — Windows One-Click Launcher:**
Double-click `run.bat` or run:
```bat
run.bat
```

**Option B — Terminal:**
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 📂 Project Structure

```text
.
├── app.py                      # Flask web server & streaming API endpoints
├── index.html                  # Single-page web chat application
├── query_data.py               # RAG pipeline, Chroma retrieval & Ollama routing
├── populate_database.py        # PDF loader, recursive chunking & vector indexing
├── get_embedding_function.py   # Embedding setup with nomic-embed-text
├── run.bat                     # Windows quick-launch script
├── requirements.txt            # Python dependencies
├── .env.example                # Sample environment variables configuration
├── .gitignore                  # Git ignore rules (prevents tracking venv & data)
└── data/                       # Directory for knowledge base PDF files
```

---

## 💬 Usage Examples

Try asking questions specific to your indexed documents:
- *"What are the key responsibilities described in the document?"*
- *"Summarize the primary guidelines on page 4."*
- *"What are the safety requirements and protocols?"*

Expand the **View Source Context** toggle underneath any bot response to verify which chunks and pages were used to construct the answer.

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
