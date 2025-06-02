# Local Flow

A minimal, local, GPU-accelerated RAG server that actually works. Ships with more dependencies than the Vatican's import list.

## Quick Start

Because reading documentation is *so* 2022.

### 1. Install Dependencies

```bash
# Create virtual environment (shocking, I know)
python3 -m venv flow-env
source flow-env/bin/activate  # Windows: .\flow-env\Scripts\activate

# Install everything (except your patience)
pip install sentence-transformers langchain-community langchain-text-splitters faiss-cpu pdfplumber requests beautifulsoup4 gitpython nbformat pydantic fastmcp

# For PyTorch with CUDA (check https://pytorch.org/get-started/locally/ for your version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Note**: Using `faiss-cpu` because `faiss-gpu` is apparently allergic to recent CUDA versions. Your embeddings will still use GPU. Chill.

### 2. Configure MCP in Cursor

Add this to your `~/.cursor/mcp.json` (or wherever Cursor keeps its secrets):

```json
{
  "mcpServers": {
    "LocalFlow": {
      "command": "wsl.exe",
      "args": [
        "-d", 
        "Ubuntu_2404", 
        "bash", 
        "-c", 
        "source /home/<you>/local_flow/flow-env/bin/activate && /home/<you>/local_flow/flow-env/bin/python /home/<you>/local_flow/rag_mcp_server.py"
      ],
      "env": {
        "RAG_DATA_DIR": "/path/to/rag_data_storage"
      },
      "scopes": ["rag_read", "rag_write"],
      "tools": ["add_source", "query_context", "list_sources", "remove_source"]
    }
  }
}
```

Server runs on `http://localhost:8081`. Revolutionary stuff. Adjust paths to your setup (or it won't work, unsurprisingly). 

### 3. Restart Cursor

Because restarting always fixes everything, right?

## Usage

Welcome to the future of document ingestion. No more curl commands, no more HTTP status codes. Just sweet, sweet MCP tools.

### Adding Documents

Tell Cursor to use the `add_source` tool:

**PDFs:**
- Source type: `pdf`
- Path: `/path/to/your/important-document.pdf` 
- Source ID: Whatever makes you happy

**Web Pages:**
- Source type: `webpage`  
- URL: `https://stackoverflow.com/questions/definitely-not-copy-pasted`
- Source ID: `web_wisdom` or whatever

**Git Repositories:**
- Source type: `git_repo`
- URL: `https://github.com/someone/hopefully-documented.git`
- Source ID: `someones_code`

The tool handles the rest. Like magic, but with more dependencies.

### Querying (who knew it could be so complicated to ask a simple question)

Use the `query_context` tool:
- Query: "What does this thing actually do?"
- Top K: How many results you want (default: 5)
- Source IDs: Filter to specific sources (optional)

### Managing Sources

- `list_sources` - See what you've fed the machine
- `remove_source` - Pretend to delete things (metadata only, embeddings stick around like bad memories)

## WSL2 Installation

Got WSL2? Lucky you. Check `INSTALL_WSL2.md` for the journey of GPU setup.

## Features

- ✅ Local execution (no cloud bills)
- ✅ GPU acceleration (when it feels like it)  
- ✅ Multiple document types (PDFs, web pages, Git repos)
- ✅ Persistent storage (remembers things between restarts)
- ✅ Source filtering (because organization matters)
- ❌ Your sanity (sold separately)

## Architecture

MCP Server + FAISS + SentenceTransformers + LangChain + FastMCP

Vector database stored in `./vector_db` (or wherever `RAG_DATA_DIR` points). Don't delete it unless you enjoy re-indexing everything.

JSON-RPC over stdin/stdout because apparently that's how we communicate with AI tools now. The future is weird.

## Troubleshooting

**"Tool not found"**: Did you restart Cursor? Restart Cursor.

**"WSL command failed"**: Check your paths in `mcp.json`. Also, is WSL actually running?

**"CUDA out of memory"**: Your GPU is having feelings. Try smaller batch sizes or less ambitious documents.

**"Permission denied"**: WSL permissions are a mystery. Try `chmod +x` on everything and pray.

**"It's not working"**: That's not a question. But yes, welcome to local AI tooling. 