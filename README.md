# Local Flow

A minimal, local, GPU-accelerated RAG server that actually works. Ships with more dependencies than the Vatican's import list. Runs on WSL2 and Windows out-of-the-box (Cursor configuration not included).

## Quick Start

Because reading documentation is *so* 2022.

### 1. Platform

- **Windows**: Native Windows setup with CUDA toolkit → See `INSTALL_WINDOWS.md` for the gory details
- **WSL2**: Linux experience on Windows → See `INSTALL_WSL2.md` for the masochistic journey

### 2. Install Dependencies

```bash
# Create virtual environment (shocking, I know)
python -m venv flow-env

# Windows
flow-env\Scripts\activate

# WSL2/Linux  
source flow-env/bin/activate

# Install everything (except your patience)
pip install sentence-transformers langchain-community langchain-text-splitters faiss-cpu pdfplumber requests beautifulsoup4 gitpython nbformat pydantic fastmcp

# PyTorch with CUDA (check https://pytorch.org/get-started/locally/ for your version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# -- CUDA 12.9 (selected 12.8 on above link) I used `cu128`
```

**Note**: Using `faiss-cpu` because `faiss-gpu` is apparently allergic to recent CUDA versions. Your embeddings will still use GPU. Chill.

### 3. Configure MCP in Cursor

Add this to your `mcp.json` file:

**Windows** (`~/.cursor/mcp.json` or `%APPDATA%\Cursor\User\globalStorage\cursor.mcp\mcp.json`):
```json
{
  "mcpServers": {
    "LocalFlow": {
      "command": "C:\\path\\to\\your\\local_flow\\flow-env\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\your\\local_flow\\rag_mcp_server.py"],
      "env": {
        "RAG_DATA_DIR": "C:\\path\\to\\your\\vector_db"
      },
      "scopes": ["rag_read", "rag_write"],
      "tools": ["add_source", "query_context", "list_sources", "remove_source"]
    }
  }
}
```

**WSL2** (`~/.cursor/mcp.json`):
```json
{
  "mcpServers": {
    "LocalFlow": {
      "command": "wsl.exe",
      "args": [
        "-d", "Ubuntu_2404", 
        "bash", "-c", 
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

### 4. Restart Cursor

Because restarting always fixes everything, right?

## Usage

Welcome to the future of document ingestion. No more curl commands, no more HTTP status codes. Just sweet, sweet MCP tools.

### Adding Documents

Tell Cursor to use the `add_source` tool:

**PDFs:**
- Source type: `pdf`
- Path: `/path/to/your/document.pdf` (Linux) or `C:\path\to\document.pdf` (Windows)
- Source ID: Whatever makes you happy

**Web Pages:**
- Source type: `webpage`  
- URL: `https://stackoverflow.com/questions/definitely-not-copy-pasted`
- Source ID: Optional

**Git Repositories:**
- Source type: `git_repo`
- URL: `https://github.com/someone/hopefully-documented.git` or local path
- Source ID: Optional identifier

The tool handles the rest. Like magic, but with more dependencies.

### Querying (who knew it could be so complicated to ask a simple question)

Use the `query_context` tool:
- Query: "What does this thing actually do?"
- Top K: How many results you want (default: 5)
- Source IDs: Filter to specific sources (optional)

### Managing Sources

- `list_sources` - See what you've fed the machine
- `remove_source` - Pretend to delete things (metadata only, embeddings stick around like bad memories)

## Features

- ✅ Local execution (no cloud bills)
- ✅ GPU acceleration (when it feels like it)  
- ✅ Multiple document types (PDFs, web pages, Git repos)
- ✅ Persistent storage (remembers things between restarts)
- ✅ Source filtering (because organization matters)
- ✅ Cross-platform (Windows native or WSL2)
- ❌ Your sanity (sold separately)

## Architecture

MCP Server + FAISS + SentenceTransformers + LangChain + FastMCP

Vector database stored in `./vector_db` (or wherever `RAG_DATA_DIR` points). Don't delete it unless you enjoy re-indexing everything.

JSON-RPC over stdin/stdout because apparently that's how we communicate with AI tools now. The future is weird.

## Troubleshooting

### Universal Issues
**"Tool not found"**: Did you restart Cursor? Restart Cursor.

**"CUDA out of memory"**: Your GPU is having feelings. Try smaller batch sizes or less ambitious documents.

**"It's not working"**: That's not a question. But yes, welcome to local AI tooling.

### Platform-Specific Issues
For detailed troubleshooting:
- **Windows**: Check `INSTALL_WINDOWS.md`
- **WSL2**: Check `INSTALL_WSL2.md`

Both have extensive troubleshooting sections because, let's face it, you'll need them. 