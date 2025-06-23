# rag_mcp_server_mcp.py

import os
import json
import shutil
import tempfile
import logging
import datetime
import sys
from typing import List, Optional, Dict, Any, Union

import torch
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
from git import Repo

# MCP specific imports
from mcp.server.fastmcp import FastMCP, Context
# REMOVED: from mcp.server.jsonrpc import ServerError, ErrorCode, MethodNotFound

# Configure logging for better visibility
# Direct logging to stderr for MCP servers to avoid mixing with stdout JSON-RPC messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stderr)

# --- Configuration Constants ---
VECTOR_DB_PATH = os.environ.get("RAG_DATA_DIR", os.path.join(os.getcwd(), "vector_db")) # Use env var for data dir, fallback to absolute path
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2" # "all-MiniLM-L6-v2" -- faster and smaller, but worse
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {DEVICE} for embedding model.")

# --- Data Models for RAG Logic ---
# These models remain largely the same, but will be used by FastMCP for type hinting and schema generation.
class AddSourceRequest(BaseModel):
    source_type: str = Field(..., description="Type of source: 'pdf', 'webpage', 'git_repo'")
    path_or_url: str = Field(..., description="Local file path, web URL, or Git repository URL")
    source_id: Optional[str] = Field(None, description="Optional unique ID for the source. If not provided, one will be generated.")

class QueryContextRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    source_ids: Optional[List[str]] = Field(None, description="Optional list of source IDs to filter the search")
    top_k: int = Field(5, description="Number of top relevant chunks to retrieve")

class RetrievedChunk(BaseModel):
    content: str
    source: str
    metadata: Dict[str, Any]

class QueryContextResponse(BaseModel):
    results: List[RetrievedChunk]

class SourceInfo(BaseModel):
    id: str
    type: str
    path_or_url: str
    num_chunks: int
    last_indexed: str

# --- RAG Server Core Logic Class ---
class RAGServer:
    def __init__(self):
        self.embedding_model = self._load_embedding_model()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        # Ensure VECTOR_DB_PATH exists before trying to load
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        self.vector_store = self._load_or_create_vector_store()
        self.indexed_sources: Dict = self._load_indexed_sources()
        logging.info(f"RAGServer initialized. Vector database will be stored at: {VECTOR_DB_PATH}")

    def _load_embedding_model(self):
        try:
            model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
            logging.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME} on {DEVICE}")
            return model
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}. Ensure PyTorch with CUDA is installed if using GPU.")

    def _get_embeddings_function(self):
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': DEVICE})

    def _load_or_create_vector_store(self):
        embeddings_function = self._get_embeddings_function()
        if os.path.exists(VECTOR_DB_PATH):
            try:
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
        metadata_file = os.path.join(VECTOR_DB_PATH, "indexed_sources.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    return {k: SourceInfo(**v) for k, v in data.items()}
            except Exception as e:
                logging.warning(f"Failed to load indexed sources metadata: {e}. Starting fresh.")
                return {}
        return {}

    def _save_indexed_sources(self):
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        metadata_file = os.path.join(VECTOR_DB_PATH, "indexed_sources.json")
        with open(metadata_file, 'w') as f:
            # Use model_dump for Pydantic V2 compatibility
            json.dump({k: v.model_dump() for k, v in self.indexed_sources.items()}, f, indent=4)
        logging.info(f"Saved indexed sources metadata to {metadata_file}")

    def _process_pdf(self, file_path: str) -> List[Document]:
        documents = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        documents.append(Document(page_content=text, metadata={"source": file_path, "page": i + 1, "source_type": "pdf"}))
            logging.info(f"Processed PDF: {file_path}, extracted {len(documents)} pages.")
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logging.error(f"Error processing PDF {file_path}: {e}")
            # Raise a standard Python exception
            raise ValueError(f"Could not process PDF: {e}")

    def _process_webpage(self, url: str) -> List[Document]:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator='\n', strip=True)

            if not text:
                logging.warning(f"No text extracted from webpage: {url}. It might be dynamic content.")
                return []

            doc = Document(page_content=text, metadata={"source": url, "source_type": "webpage"})
            logging.info(f"Processed webpage: {url}, extracted text length: {len(text)}.")
            return self.text_splitter.split_documents([doc])
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching webpage {url}: {e}")
            # Raise a standard Python exception
            raise ValueError(f"Could not fetch webpage: {e}")
        except Exception as e:
            logging.error(f"Error processing webpage {url}: {e}")
            # Raise a standard Python exception
            raise RuntimeError(f"Error processing webpage: {e}")

    def _process_git_repo(self, repo_url: str) -> List[Document]:
        documents = []
        cleanup_temp = False
        temp_dir = None
        
        try:
            # Check if this is a local path to an existing Git repository
            if os.path.exists(repo_url) and os.path.isdir(repo_url):
                # Check if it's a valid Git repository
                try:
                    repo = Repo(repo_url)
                    repo_dir = repo_url
                    logging.info(f"Using local Git repository: {repo_url}")
                except Exception as e:
                    logging.error(f"Path exists but is not a valid Git repository: {repo_url}, {e}")
                    raise ValueError(f"Invalid Git repository path: {e}")
            else:
                # It's a remote URL, clone to temp directory
                temp_dir = tempfile.mkdtemp()
                repo_dir = temp_dir
                cleanup_temp = True
                logging.info(f"Cloning Git repository: {repo_url} to {temp_dir}")
                Repo.clone_from(repo_url, temp_dir)

            relevant_extensions = ('.py', '.md', '.txt', '.rst', '.ipynb', '.json', '.xml', '.yaml', '.yml', '.sh', '.c', '.cpp', '.h', '.hpp')

            for root, _, files in os.walk(repo_dir):
                for file_name in files:
                    if file_name.endswith(relevant_extensions):
                        file_path = os.path.join(root, file_name)
                        relative_path = os.path.relpath(file_path, repo_dir)
                        try:
                            content = ""
                            if file_name.endswith('.ipynb'):
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    notebook_content = nbformat.read(f, as_version=4)
                                    for cell in notebook_content.cells:
                                        if cell.cell_type in ['code', 'markdown']:
                                            content += cell.source + "\n\n"
                                logging.debug(f"Processed .ipynb file: {relative_path}")
                            else:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                logging.debug(f"Processed text file: {relative_path}")

                            if content:
                                documents.append(Document(page_content=content, metadata={"source": repo_url, "file_path": relative_path, "source_type": "git_repo"}))
                        except Exception as e:
                            logging.warning(f"Could not read/process file {file_path}: {e}")

            logging.info(f"Processed Git repo: {repo_url}, extracted {len(documents)} documents.")
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logging.error(f"Error processing Git repository {repo_url}: {e}")
            # Raise a standard Python exception
            raise ValueError(f"Could not process Git repository: {e}")
        finally:
            if cleanup_temp and temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logging.info(f"Cleaned up temporary directory: {temp_dir}")

    def add_source(self, request: AddSourceRequest):
        source_id = request.source_id if request.source_id else f"{request.source_type}_{len(self.indexed_sources) + 1}"
        if source_id in self.indexed_sources:
            # Raise a standard Python exception
            raise ValueError(f"Source ID '{source_id}' already exists. Please choose a different one or remove the existing source.")

        documents_to_add = []
        try:
            logging.info(f"Starting to process source: {request.source_type} - {request.path_or_url}")
            
            if request.source_type == "pdf":
                # Check if file exists first
                if not os.path.exists(request.path_or_url):
                    raise ValueError(f"PDF file not found: {request.path_or_url}")
                documents_to_add = self._process_pdf(request.path_or_url)
            elif request.source_type == "webpage":
                documents_to_add = self._process_webpage(request.path_or_url)
            elif request.source_type == "git_repo":
                documents_to_add = self._process_git_repo(request.path_or_url)
            else:
                # Raise a standard Python exception
                raise ValueError("Invalid source_type. Must be 'pdf', 'webpage', or 'git_repo'.")

            if not documents_to_add:
                # Raise a standard Python exception
                raise ValueError(f"No content extracted from source: {request.path_or_url}")

            logging.info(f"Successfully processed {len(documents_to_add)} document chunks")

            for doc in documents_to_add:
                doc.metadata["source_id"] = source_id

            logging.info(f"Adding documents to vector store...")
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents_to_add, self._get_embeddings_function())
                logging.info(f"Created new FAISS vector store with {len(documents_to_add)} documents.")
            else:
                self.vector_store.add_documents(documents_to_add)
                logging.info(f"Added {len(documents_to_add)} documents to existing vector store.")

            logging.info(f"Saving vector store to {VECTOR_DB_PATH}")
            self.vector_store.save_local(VECTOR_DB_PATH)

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

        except (ValueError, RuntimeError): # Catch the exceptions raised by _process_ functions
            raise # Re-raise them to be caught by FastMCP
        except Exception as e:
            logging.error(f"Failed to add source {request.path_or_url}: {e}", exc_info=True)
            # Raise a standard Python exception for unexpected errors
            raise RuntimeError(f"Failed to add source: {e}")

    def rebuild_index(self):
        """
        Completely rebuilds the FAISS index from scratch using all currently indexed sources.
        This is useful when the index becomes corrupted or when you want to remove old embeddings.
        """
        logging.info("Starting complete index rebuild...")
        
        if not self.indexed_sources:
            logging.info("No sources to rebuild index from.")
            return {"status": "success", "message": "No sources found, index cleared."}
        
        # Clear the current vector store
        self.vector_store = None
        
        # Temporarily store source info
        sources_to_rebuild = list(self.indexed_sources.items())
        self.indexed_sources.clear()
        
        rebuild_results = []
        for source_id, source_info in sources_to_rebuild:
            try:
                logging.info(f"Re-indexing source: {source_id}")
                request = AddSourceRequest(
                    source_type=source_info.type,
                    path_or_url=source_info.path_or_url,
                    source_id=source_id
                )
                result = self.add_source(request)
                rebuild_results.append(f"✓ {source_id}: {result['num_chunks_indexed']} chunks")
            except Exception as e:
                logging.error(f"Failed to re-index source {source_id}: {e}")
                rebuild_results.append(f"✗ {source_id}: FAILED - {str(e)}")
        
        logging.info("Index rebuild completed")
        return {
            "status": "success", 
            "message": f"Rebuilt index with {len(self.indexed_sources)} sources",
            "details": rebuild_results
        }

    def clear_all_sources(self):
        """
        Completely clears all sources and rebuilds an empty index.
        """
        logging.info("Clearing all sources and rebuilding empty index...")
        
        # Clear metadata
        self.indexed_sources.clear()
        self._save_indexed_sources()
        
        # Clear vector store
        self.vector_store = None
        
        # Remove the FAISS index files
        if os.path.exists(VECTOR_DB_PATH):
            for file in os.listdir(VECTOR_DB_PATH):
                if file.startswith('index'):
                    file_path = os.path.join(VECTOR_DB_PATH, file)
                    try:
                        os.remove(file_path)
                        logging.info(f"Removed FAISS index file: {file}")
                    except Exception as e:
                        logging.warning(f"Could not remove {file_path}: {e}")
        
        logging.info("All sources cleared and index reset")
        return {"status": "success", "message": "All sources cleared and index reset"}

    def query_context(self, request: QueryContextRequest) -> QueryContextResponse:
        try:
            if self.vector_store is None:
                logging.info("No documents indexed yet. Returning empty results.")
                return QueryContextResponse(results=[])

            filter_dict = {}
            if request.source_ids:
                for sid in request.source_ids:
                    if sid not in self.indexed_sources:
                        # Raise a standard Python exception
                        raise ValueError(f"Source ID '{sid}' not found.")
                filter_dict["source_id"] = {"$in": request.source_ids}

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
        except ValueError: # Catch the exception raised above
            raise # Re-raise it to be caught by FastMCP
        except Exception as e:
            logging.error(f"Failed to query context: {e}")
            # Raise a standard Python exception for unexpected errors
            raise RuntimeError(f"Failed to query context: {e}")

    def list_sources(self) -> List[SourceInfo]:
        return list(self.indexed_sources.values())

    def remove_source(self, source_id: str):
        if source_id not in self.indexed_sources:
            # Raise a standard Python exception
            raise ValueError(f"Source ID '{source_id}' not found.")

        del self.indexed_sources[source_id]
        self._save_indexed_sources()
        logging.warning(f"Source '{source_id}' removed from metadata. Note: Its embeddings remain in the FAISS index. Consider rebuilding the index for full removal if space is a concern.")
        return {"status": "success", "message": f"Source '{source_id}' metadata removed. Full removal requires re-indexing."}

# --- FastMCP Application Setup ---
# Initialize FastMCP with a name for your server
mcp_app = FastMCP("Local RAG")

# Initialize the RAG server instance
rag_server_instance = RAGServer()

# Register RAG server methods as MCP tools
# The docstrings and type hints are used by FastMCP to generate the tool schema.
@mcp_app.tool("add_source")
def add_source_tool(request: AddSourceRequest) -> Dict[str, Union[str, int]]:
    """
    Adds a new data source (PDF, webpage, or Git repository) to the RAG index.
    This tool initiates the ingestion, chunking, embedding, and storage process.
    """
    return rag_server_instance.add_source(request)

@mcp_app.tool("query_context")
def query_context_tool(request: QueryContextRequest) -> QueryContextResponse:
    """
    Queries the RAG index for relevant context based on a natural language query.
    Optionally filters the search to specific indexed source IDs.
    The retrieved context is returned as an MCP "Resource".
    """
    return rag_server_instance.query_context(request)

@mcp_app.tool("list_sources")
def list_sources_tool() -> List[SourceInfo]:
    """
    Lists all currently indexed data sources, providing their IDs, types, and other metadata.
    """
    return rag_server_instance.list_sources()

@mcp_app.tool("remove_source")
def remove_source_tool(source_id: str) -> Dict[str, str]:
    """
    Removes a specific data source from the RAG index metadata.
    Note: Due to FAISS's design, this operation primarily removes the source's metadata.
    The actual embeddings will remain in the FAISS index until a full re-index.
    """
    return rag_server_instance.remove_source(source_id)

@mcp_app.tool("rebuild_index")
def rebuild_index_tool() -> Dict[str, Union[str, List[str]]]:
    """
    Completely rebuilds the FAISS index from scratch using all currently indexed sources.
    This is useful when the index becomes corrupted or when old embeddings need to be removed.
    """
    return rag_server_instance.rebuild_index()

@mcp_app.tool("clear_all_sources")
def clear_all_sources_tool() -> Dict[str, str]:
    """
    Completely clears all sources and rebuilds an empty index.
    This removes all metadata and FAISS index files for a clean start.
    """
    return rag_server_instance.clear_all_sources()

# --- Main execution block for MCP Server ---
if __name__ == "__main__":
    # FastMCP will automatically handle reading from stdin and writing to stdout
    mcp_app.run()