# rag_mcp_server.py

import os
import json
import shutil
import tempfile
import logging
import datetime
from typing import List, Optional, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Core RAG Libraries
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Specific Parsers
import pdfplumber
import requests
from bs4 import BeautifulSoup
import nbformat
from git import Repo # Using gitpython for cloning Git repositories

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Constants ---
# Directory to store the FAISS vector database and source metadata
VECTOR_DB_PATH = "./vector_db"
# Name of the Sentence Transformer model for embeddings.
# 'all-MiniLM-L6-v2' is lightweight and efficient. 'all-mpnet-base-v2' is more performant but larger.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
# Chunk size for text splitting (in characters/tokens)
CHUNK_SIZE = 1000
# Overlap between consecutive chunks to preserve context
CHUNK_OVERLAP = 200

# Determine the device for the embedding model (GPU if available, otherwise CPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {DEVICE} for embedding model.")

# --- Data Models for FastAPI Endpoints ---
class AddSourceRequest(BaseModel):
    """Request model for adding a new data source."""
    source_type: str = Field(..., description="Type of source: 'pdf', 'webpage', 'git_repo'")
    path_or_url: str = Field(..., description="Local file path, web URL, or Git repository URL")
    source_id: Optional[str] = Field(None, description="Optional unique ID for the source. If not provided, one will be generated.")

class QueryContextRequest(BaseModel):
    """Request model for querying the RAG index."""
    query: str = Field(..., description="Natural language query")
    source_ids: Optional[List[str]] = Field(None, description="Optional list of source IDs to filter the search")
    top_k: int = Field(5, description="Number of top relevant chunks to retrieve")

class RetrievedChunk(BaseModel):
    """Response model for a single retrieved chunk of text."""
    content: str
    source: str
    metadata: Dict[str, Any]

class QueryContextResponse(BaseModel):
    """Response model for a context query, containing a list of retrieved chunks."""
    results: List

class SourceInfo(BaseModel):
    """Model to store metadata about an indexed source."""
    id: str
    type: str
    path_or_url: str
    num_chunks: int
    last_indexed: str

# --- RAG Server Core Logic Class ---
class RAGServer:
    def __init__(self):
        """Initializes the RAG server, loading models and vector store."""
        self.embedding_model = self._load_embedding_model()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""] # Prioritize splitting by paragraphs, then lines, then words [1, 2]
        )
        self.vector_store = self._load_or_create_vector_store()
        self.indexed_sources: Dict = self._load_indexed_sources()

    def _load_embedding_model(self):
        """Loads the Sentence Transformer embedding model onto the specified device."""
        try:
            # Explicitly set device to 'cuda' for GPU acceleration [3, 4]
            model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
            logging.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME} on {DEVICE}")
            return model
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}. Ensure PyTorch with CUDA is installed if using GPU.")

    def _get_embeddings_function(self):
        """Returns a HuggingFaceEmbeddings instance for LangChain integration."""
        # LangChain's HuggingFaceEmbeddings wraps SentenceTransformer [5]
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': DEVICE})

    def _load_or_create_vector_store(self):
        """Loads an existing FAISS vector store or returns None if not found."""
        embeddings_function = self._get_embeddings_function()
        if os.path.exists(VECTOR_DB_PATH):
            try:
                # Load local FAISS index. allow_dangerous_deserialization is needed for security reasons
                # when loading from untrusted sources, but is fine for local use.
                vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings_function, allow_dangerous_deserialization=True)
                logging.info(f"Loaded existing FAISS vector store from {VECTOR_DB_PATH}")
                return vector_store
            except Exception as e:
                logging.warning(f"Failed to load FAISS index from {VECTOR_DB_PATH}: {e}. Will create new one when first document is added.")
                return None
        else:
            logging.info(f"No existing FAISS vector store found. Will create one when first document is added.")
            return None

    def _load_indexed_sources(self) -> Dict:
        """Loads metadata about indexed sources from a JSON file for persistence."""
        metadata_file = os.path.join(VECTOR_DB_PATH, "indexed_sources.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    # Reconstruct SourceInfo objects from dictionary data
                    return {k: SourceInfo(**v) for k, v in data.items()}
            except Exception as e:
                logging.warning(f"Failed to load indexed sources metadata: {e}. Starting fresh.")
                return {}
        return {}

    def _save_indexed_sources(self):
        """Saves metadata about indexed sources to a JSON file."""
        os.makedirs(VECTOR_DB_PATH, exist_ok=True) # Ensure directory exists
        metadata_file = os.path.join(VECTOR_DB_PATH, "indexed_sources.json")
        with open(metadata_file, 'w') as f:
            # Convert SourceInfo objects to dictionaries for JSON serialization
            json.dump({k: v.dict() for k, v in self.indexed_sources.items()}, f, indent=4)
        logging.info(f"Saved indexed sources metadata to {metadata_file}")

    def _process_pdf(self, file_path: str) -> List:
        """
        Loads and chunks text from a PDF file using pdfplumber.
        Prioritizes layout preservation for technical documents.[6, 2]
        """
        documents = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        # Add metadata for source tracking [1]
                        documents.append(Document(page_content=text, metadata={"source": file_path, "page": i + 1, "source_type": "pdf"}))
            logging.info(f"Processed PDF: {file_path}, extracted {len(documents)} pages.")
            return self.text_splitter.split_documents(documents) # Chunk the extracted text [7, 1]
        except Exception as e:
            logging.error(f"Error processing PDF {file_path}: {e}")
            raise HTTPException(status_code=422, detail=f"Could not process PDF: {e}")

    def _process_webpage(self, url: str) -> List:
        """
        Loads and chunks text from a web page using requests and BeautifulSoup.
        Note: This minimal implementation is best for static web pages.
        Dynamic (JavaScript-rendered) content may not be fully captured.[8]
        """
        try:
            response = requests.get(url)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements to get clean text [9]
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator='\n', strip=True)
            
            if not text:
                logging.warning(f"No text extracted from webpage: {url}. It might be dynamic content.")
                return []

            doc = Document(page_content=text, metadata={"source": url, "source_type": "webpage"})
            logging.info(f"Processed webpage: {url}, extracted text length: {len(text)}.")
            return self.text_splitter.split_documents([doc]) # Chunk the extracted text [7, 1]
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching webpage {url}: {e}")
            raise HTTPException(status_code=422, detail=f"Could not fetch webpage: {e}")
        except Exception as e:
            logging.error(f"Error processing webpage {url}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing webpage: {e}")

    def _process_git_repo(self, repo_url: str) -> List:
        """
        Clones a Git repository and processes relevant files (code, markdown, Jupyter notebooks).
        Uses gitpython for cloning and nbformat for.ipynb files.[10, 11]
        """
        temp_dir = tempfile.mkdtemp()
        documents = []
        try:
            logging.info(f"Cloning Git repository: {repo_url} to {temp_dir}")
            Repo.clone_from(repo_url, temp_dir) # Clone the repository

            # Define relevant file extensions for indexing [10]
            relevant_extensions = ('.py', '.md', '.txt', '.rst', '.ipynb', '.json', '.xml', '.yaml', '.yml', '.sh', '.c', '.cpp', '.h', '.hpp')
            
            for root, _, files in os.walk(temp_dir):
                for file_name in files:
                    if file_name.endswith(relevant_extensions):
                        file_path = os.path.join(root, file_name)
                        relative_path = os.path.relpath(file_path, temp_dir) # Get path relative to repo root
                        try:
                            content = ""
                            if file_name.endswith('.ipynb'):
                                # Parse Jupyter notebooks to extract code and markdown cells [11]
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    notebook_content = nbformat.read(f, as_version=4)
                                    for cell in notebook_content.cells:
                                        if cell.cell_type in ['code', 'markdown']:
                                            content += cell.source + "\n\n" # Concatenate cell content
                                logging.debug(f"Processed.ipynb file: {relative_path}")
                            else:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                logging.debug(f"Processed text file: {relative_path}")

                            if content:
                                # Add metadata for source tracking [7]
                                documents.append(Document(page_content=content, metadata={"source": repo_url, "file_path": relative_path, "source_type": "git_repo"}))
                        except Exception as e:
                            logging.warning(f"Could not read/process file {file_path}: {e}")
            
            logging.info(f"Processed Git repo: {repo_url}, extracted {len(documents)} documents.")
            return self.text_splitter.split_documents(documents) # Chunk the extracted text [7, 1]
        except Exception as e:
            logging.error(f"Error processing Git repository {repo_url}: {e}")
            raise HTTPException(status_code=422, detail=f"Could not process Git repository: {e}")
        finally:
            # Clean up the temporary directory where the repo was cloned
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logging.info(f"Cleaned up temporary directory: {temp_dir}")

    def add_source(self, request: AddSourceRequest):
        """
        Adds a new data source (PDF, webpage, or Git repository) to the RAG index.
        Generates embeddings and persists the updated vector store.
        """
        source_id = request.source_id if request.source_id else f"{request.source_type}_{len(self.indexed_sources) + 1}"
        if source_id in self.indexed_sources:
            raise HTTPException(status_code=400, detail=f"Source ID '{source_id}' already exists. Please choose a different one or remove the existing source.")

        documents_to_add = []
        try:
            if request.source_type == "pdf":
                documents_to_add = self._process_pdf(request.path_or_url)
            elif request.source_type == "webpage":
                documents_to_add = self._process_webpage(request.path_or_url)
            elif request.source_type == "git_repo":
                documents_to_add = self._process_git_repo(request.path_or_url)
            else:
                raise HTTPException(status_code=400, detail="Invalid source_type. Must be 'pdf', 'webpage', or 'git_repo'.")

            if not documents_to_add:
                raise HTTPException(status_code=400, detail=f"No content extracted from source: {request.path_or_url}")

            # Add source_id to each document's metadata for filtering [III]
            for doc in documents_to_add:
                doc.metadata["source_id"] = source_id

            # Add documents to the vector store and save it [7, 5]
            if self.vector_store is None:
                # Create the vector store with the first batch of documents
                self.vector_store = FAISS.from_documents(documents_to_add, self._get_embeddings_function())
                logging.info(f"Created new FAISS vector store with {len(documents_to_add)} documents.")
            else:
                # Add documents to existing vector store
                self.vector_store.add_documents(documents_to_add)
            
            self.vector_store.save_local(VECTOR_DB_PATH) # Persist the updated index

            # Update indexed sources metadata for tracking
            self.indexed_sources[source_id] = SourceInfo(
                id=source_id,
                type=request.source_type,
                path_or_url=request.path_or_url,
                num_chunks=len(documents_to_add),
                last_indexed=datetime.datetime.now().isoformat()
            )
            self._save_indexed_sources()

            logging.info(f"Successfully added source '{source_id}' with {len(documents_to_add)} chunks.")
            return {"status": "success", "source_id": source_id, "num_chunks_indexed": len(documents_to_add)}

        except HTTPException:
            raise # Re-raise FastAPI HTTPExceptions
        except Exception as e:
            logging.error(f"Failed to add source {request.path_or_url}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to add source: {e}")

    def query_context(self, request: QueryContextRequest) -> QueryContextResponse:
        """
        Queries the RAG index for relevant context based on a natural language query.
        Supports filtering by specific source IDs [III].
        """
        try:
            # Check if vector store exists
            if self.vector_store is None:
                logging.info("No documents indexed yet. Returning empty results.")
                return QueryContextResponse(results=[])

            # Build filter based on source_ids for precise retrieval [III]
            filter_dict = {}
            if request.source_ids:
                # Ensure all requested source_ids exist
                for sid in request.source_ids:
                    if sid not in self.indexed_sources:
                        raise HTTPException(status_code=404, detail=f"Source ID '{sid}' not found.")
                # LangChain FAISS filter syntax for multiple source_ids
                filter_dict["source_id"] = {"$in": request.source_ids} 

            # Perform similarity search with filtering [5]
            retrieved_docs = self.vector_store.similarity_search(
                query=request.query,
                k=request.top_k,
                filter=filter_dict if filter_dict else None
            )

            results = [
                RetrievedChunk(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    metadata=doc.metadata
                ) for doc in retrieved_docs
            ]
            logging.info(f"Query '{request.query}' retrieved {len(results)} chunks.")
            return QueryContextResponse(results=results)
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Failed to query context: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to query context: {e}")

    def list_sources(self) -> List:
        """Lists all currently indexed data sources and their metadata."""
        return list(self.indexed_sources.values())

    def remove_source(self, source_id: str):
        """
        Removes a specific data source from the RAG index metadata.
        NOTE: Due to FAISS's design, this only removes the source from our metadata.
        The actual embeddings remain in the FAISS index until a full re-index or
        a new index is created. For a truly "removed" source, the index would need
        to be rebuilt from remaining sources, which is a computationally intensive operation.
        """
        if source_id not in self.indexed_sources:
            raise HTTPException(status_code=404, detail=f"Source ID '{source_id}' not found.")
        
        del self.indexed_sources[source_id]
        self._save_indexed_sources()
        logging.warning(f"Source '{source_id}' removed from metadata. Note: Its embeddings remain in the FAISS index. Consider rebuilding the index for full removal if space is a concern.")
        return {"status": "success", "message": f"Source '{source_id}' metadata removed. Full removal requires re-indexing."}

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Local RAG MCP Server",
    description="A minimal, local, GPU-accelerated RAG server for code browser integration via Model Context Protocol.",
    version="1.0.0"
)

# Initialize the RAG server instance
rag_server = RAGServer()

# --- FastAPI Endpoints (MCP Tools) ---
@app.post("/add_source")
async def add_source_endpoint(request: AddSourceRequest):
    """
    **MCP Tool: `add_source`**
    Adds a new data source (PDF, webpage, or Git repository) to the RAG index.
    This tool initiates the ingestion, chunking, embedding, and storage process.
    """
    return rag_server.add_source(request)

@app.post("/query_context", response_model=QueryContextResponse)
async def query_context_endpoint(request: QueryContextRequest):
    """
    **MCP Tool: `query_context`**
    Queries the RAG index for relevant context based on a natural language query.
    Optionally filters the search to specific indexed source IDs.
    The retrieved context is returned as an MCP "Resource".
    """
    return rag_server.query_context(request)

@app.get("/list_sources", response_model=List)
async def list_sources_endpoint():
    """
    **MCP Tool: `list_sources`**
    Lists all currently indexed data sources, providing their IDs, types, and other metadata.
    """
    return rag_server.list_sources()

@app.delete("/remove_source/{source_id}")
async def remove_source_endpoint(source_id: str):
    """
    **MCP Tool: `remove_source`**
    Removes a specific data source from the RAG index metadata.
    Note: Due to FAISS's design, this operation primarily removes the source's metadata.
    The actual embeddings will remain in the FAISS index until a full re-index.
    """
    return rag_server.remove_source(source_id)

# --- Main execution block for Uvicorn (HTTP Server) ---
if __name__ == "__main__":
    import uvicorn
    # Create the directory for the vector database if it doesn't exist
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    logging.info(f"Vector database will be stored at: {VECTOR_DB_PATH}")
    
    # Run the FastAPI application using Uvicorn.
    # Host '0.0.0.0' makes it accessible from other applications on the network (e.g., Cursor).
    # Port 8081 is a common choice for local services.
    uvicorn.run(app, host="0.0.0.0", port=8081)
