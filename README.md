# Local Flow

A minimal, local, GPU-accelerated RAG server that actually works. Ships with more dependencies than the Vatican's import list.

## Quick Start

Because reading documentation is *so* 2022.

### 1. Install Dependencies

```bash
# Create virtual environment (shocking, I know)
python3 -m venv flow-env
source flow-env/bin/activate  # Windows: .\flow-env\Scripts\activate

# Install everything (except CUDA)
pip install fastapi uvicorn sentence-transformers langchain-community langchain-text-splitters faiss-cpu pdfplumber requests beautifulsoup4 gitpython nbformat pydantic

# For PyTorch with CUDA (check https://pytorch.org/get-started/locally/ for your version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Note**: Using `faiss-cpu` because `faiss-gpu` is apparently allergic to recent CUDA versions. Your embeddings will still use GPU. Chill.

### 2. Run the Server

```bash
python rag_mcp_server.py
```

Server runs on `http://localhost:8081`. Revolutionary stuff.

## Adding Documents

### PDF Files
```bash
curl -X POST "http://localhost:8081/add_source" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "pdf", 
    "path_or_url": "/path/to/document.pdf",
    "source_id": "my_important_pdf"
  }'
```

### Web Pages
```bash
curl -X POST "http://localhost:8081/add_source" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "webpage", 
    "path_or_url": "https://example.com/definitely-not-stackoverflow",
    "source_id": "web_wisdom"
  }'
```

### Git Repositories
```bash
curl -X POST "http://localhost:8081/add_source" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "git_repo", 
    "path_or_url": "https://github.com/user/repo.git",
    "source_id": "someones_code"
  }'
```

## Querying Your New Digital Brain

```bash
curl -X POST "http://localhost:8081/query_context" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does this even do?",
    "top_k": 5
  }'
```

## Other Useful Endpoints

- `GET /list_sources` - See what you've fed the machine
- `DELETE /remove_source/{source_id}` - Pretend to delete things
- `GET /docs` - Interactive API docs at `http://localhost:8081/docs`

## WSL2 Installation

Got WSL2? Lucky you. Check `INSTALL_WSL2.md` for the *delightful* journey of GPU setup.

## Features

- ✅ Local execution (no cloud bills)
- ✅ GPU acceleration (when it feels like it)  
- ✅ Multiple document types (PDFs, web pages, Git repos)
- ✅ Persistent storage (remembers things between restarts)
- ✅ Source filtering (because organization matters)
- ❌ Your sanity (sold separately)

## Architecture

FastAPI server + FAISS + SentenceTransformers + LangChain

Vector database stored in `./vector_db`. Don't delete it unless you enjoy re-indexing everything. 