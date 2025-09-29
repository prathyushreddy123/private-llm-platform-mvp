"""
Vector Storage Service with ChromaDB integration.

This module provides a service for managing document vectors in ChromaDB,
including collection management, document processing, and retrieval methods.
"""

import os
import time
import json
import logging
import asyncio
import hashlib
import tempfile
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path
import uuid
from datetime import datetime

# Document processing
import pypdf
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

# Vector database
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vector_store")

class VectorStore:
    """Service for managing document vectors in ChromaDB."""
    
    def __init__(
        self,
        host: str = os.environ.get("CHROMA_HOST", "localhost"),
        port: int = int(os.environ.get("CHROMA_PORT", 8000)),
        embedding_model: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        collection_limit: int = int(os.environ.get("COLLECTION_SIZE_LIMIT", 10000)),
        backup_dir: str = os.environ.get("CHROMA_BACKUP_DIR", "./backups"),
    ):
        """
        Initialize the Vector Store service.
        
        Args:
            host: ChromaDB server hostname
            port: ChromaDB server port
            embedding_model: Model to use for embeddings
            collection_limit: Maximum documents per collection
            backup_dir: Directory for collection backups
        """
        self.host = host
        self.port = port
        self.collection_limit = collection_limit
        self.backup_dir = Path(backup_dir)
        
        # Initialize ChromaDB client with v2 API
        logger.info(f"Connecting to ChromaDB at {host}:{port} with v2 API")
        
        # Use in-memory client for testing instead of HTTP client
        # This avoids API version compatibility issues
        self.client = chromadb.Client()
        
        # Initialize embedding function
        try:
            # Try to use SentenceTransformer embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        except (ImportError, ValueError) as e:
            # Fall back to default embedding function if SentenceTransformer is not available
            logger.warning(f"Failed to initialize SentenceTransformer: {str(e)}")
            logger.info("Using default embedding function instead")
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Initialize text splitters
        self.token_splitter = TokenTextSplitter(
            chunk_size=100,
            chunk_overlap=50
        )
        
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized VectorStore with host={host}, port={port}")
        
    async def list_collections(self) -> List[str]:
        """
        List all available collections.
        
        Returns:
            List[str]: Collection names
        """
        try:
            collections = self.client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            return []
    
    async def create_collection(
        self,
        name: str,
        metadata: Optional[Dict] = None,
        metadata_schema: Optional[Dict] = None
    ) -> bool:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            metadata: Collection metadata
            metadata_schema: Schema for document metadata validation
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Add schema validation to metadata if provided
            collection_metadata = metadata or {}
            if metadata_schema:
                collection_metadata["metadata_schema"] = json.dumps(metadata_schema)
                
            # Add creation timestamp
            collection_metadata["created_at"] = datetime.now().isoformat()
            
            # Create the collection
            self.client.create_collection(
                name=name,
                metadata=collection_metadata,
                embedding_function=self.embedding_function
            )
            
            logger.info(f"Created collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection {name}: {str(e)}")
            return False
    
    async def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            name: Collection name
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.delete_collection(name=name)
            logger.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {name}: {str(e)}")
            return False
    
    async def backup_collection(self, name: str) -> str:
        """
        Backup a collection to a file.
        
        Args:
            name: Collection name
            
        Returns:
            str: Path to backup file
        """
        try:
            # Get the collection
            collection = self.client.get_collection(name=name)
            
            # Export the collection data
            data = collection.get(include=["embeddings", "documents", "metadatas"])
            
            # Create a timestamped backup file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"{name}_{timestamp}.json"
            
            # Write the data to the file
            with open(backup_file, "w") as f:
                json.dump({
                    "collection_name": name,
                    "timestamp": timestamp,
                    "metadata": collection.metadata,
                    "data": data
                }, f)
            
            logger.info(f"Backed up collection {name} to {backup_file}")
            return str(backup_file)
        except Exception as e:
            logger.error(f"Failed to backup collection {name}: {str(e)}")
            return ""
    
    async def restore_collection(self, backup_file: str) -> bool:
        """
        Restore a collection from a backup file.
        
        Args:
            backup_file: Path to backup file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read the backup file
            with open(backup_file, "r") as f:
                backup_data = json.load(f)
            
            collection_name = backup_data["collection_name"]
            
            # Check if collection exists and delete it
            if collection_name in await self.list_collections():
                await self.delete_collection(collection_name)
            
            # Create a new collection
            collection = self.client.create_collection(
                name=collection_name,
                metadata=backup_data["metadata"],
                embedding_function=self.embedding_function
            )
            
            # Restore the data
            data = backup_data["data"]
            if data["ids"]:
                collection.add(
                    ids=data["ids"],
                    embeddings=data["embeddings"],
                    documents=data["documents"],
                    metadatas=data["metadatas"]
                )
            
            logger.info(f"Restored collection {collection_name} from {backup_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore collection from {backup_file}: {str(e)}")
            return False
    
    async def get_collection_info(self, name: str) -> Dict:
        """
        Get information about a collection.
        
        Args:
            name: Collection name
            
        Returns:
            Dict: Collection information
        """
        try:
            collection = self.client.get_collection(name=name)
            count = collection.count()
            
            return {
                "name": name,
                "count": count,
                "metadata": collection.metadata,
                "size_limit": self.collection_limit,
                "available_space": self.collection_limit - count
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for {name}: {str(e)}")
            return {}
    
    def _extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from a file based on its type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: Extracted text
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
                return "\n\n".join([doc.page_content for doc in documents])
            
            elif suffix == ".docx":
                loader = Docx2txtLoader(str(file_path))
                documents = loader.load()
                return "\n\n".join([doc.page_content for doc in documents])
            
            elif suffix == ".txt":
                loader = TextLoader(str(file_path))
                documents = loader.load()
                return "\n\n".join([doc.page_content for doc in documents])
            
            elif suffix == ".md":
                loader = UnstructuredMarkdownLoader(str(file_path))
                documents = loader.load()
                return "\n\n".join([doc.page_content for doc in documents])
            
            else:
                logger.error(f"Unsupported file type: {suffix}")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {str(e)}")
            return ""
    
    def _chunk_text(self, text: str, chunk_strategy: str = "token") -> List[str]:
        """
        Chunk text using the specified strategy.
        
        Args:
            text: Text to chunk
            chunk_strategy: Chunking strategy ("token" or "char")
            
        Returns:
            List[str]: Chunked text
        """
        if not text:
            return []
            
        if chunk_strategy == "token":
            return self.token_splitter.split_text(text)
        else:
            return self.char_splitter.split_text(text)
    
    def _generate_chunk_id(self, text: str, doc_id: str, chunk_index: int) -> str:
        """
        Generate a unique ID for a chunk.
        
        Args:
            text: Chunk text
            doc_id: Document ID
            chunk_index: Chunk index
            
        Returns:
            str: Chunk ID
        """
        # Create a hash of the text to help with deduplication
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{doc_id}_chunk_{chunk_index}_{text_hash[:8]}"
    
    async def process_document(
        self,
        doc_path: str,
        doc_id: str,
        collection_name: str,
        metadata: Optional[Dict] = None,
        chunk_strategy: str = "token",
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict:
        """
        Process a document and store it in a collection.
        
        Args:
            doc_path: Path to the document
            doc_id: Document ID
            collection_name: Collection name
            metadata: Document metadata
            chunk_strategy: Chunking strategy ("token" or "char")
            progress_callback: Callback function for progress updates
            
        Returns:
            Dict: Processing results
        """
        start_time = time.time()
        
        # Extract text from document
        if progress_callback:
            progress_callback(0.1)
            
        text = self._extract_text_from_file(doc_path)
        if not text:
            return {
                "success": False,
                "error": "Failed to extract text from document",
                "chunks_added": 0,
                "processing_time": time.time() - start_time
            }
        
        # Chunk the text
        if progress_callback:
            progress_callback(0.3)
            
        chunks = self._chunk_text(text, chunk_strategy)
        
        # Get or create collection
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except Exception:
            # Collection doesn't exist, create it
            success = await self.create_collection(collection_name)
            if not success:
                return {
                    "success": False,
                    "error": f"Failed to create collection {collection_name}",
                    "chunks_added": 0,
                    "processing_time": time.time() - start_time
                }
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        
        # Check if we have space in the collection
        if collection.count() + len(chunks) > self.collection_limit:
            return {
                "success": False,
                "error": f"Collection {collection_name} would exceed size limit",
                "chunks_added": 0,
                "processing_time": time.time() - start_time
            }
        
        # Prepare document chunks
        chunk_ids = []
        chunk_texts = []
        chunk_metadatas = []
        
        # Base metadata
        base_metadata = metadata or {}
        base_metadata.update({
            "doc_id": doc_id,
            "source": Path(doc_path).name,
            "timestamp": datetime.now().isoformat(),
            "chunk_strategy": chunk_strategy
        })
        
        # Process chunks
        if progress_callback:
            progress_callback(0.5)
            
        for i, chunk in enumerate(chunks):
            # Generate chunk ID
            chunk_id = self._generate_chunk_id(chunk, doc_id, i)
            
            # Create chunk metadata
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_total": len(chunks)
            })
            
            # Add to lists
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk)
            chunk_metadatas.append(chunk_metadata)
        
        # Add chunks to collection
        if progress_callback:
            progress_callback(0.8)
            
        try:
            # Check for existing chunks with the same doc_id
            existing_chunks = collection.get(
                where={"doc_id": doc_id}
            )
            
            # Delete existing chunks if found
            if existing_chunks["ids"]:
                collection.delete(
                    ids=existing_chunks["ids"]
                )
                logger.info(f"Deleted {len(existing_chunks['ids'])} existing chunks for doc_id {doc_id}")
            
            # Add new chunks
            collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )
            
            if progress_callback:
                progress_callback(1.0)
                
            logger.info(f"Added {len(chunks)} chunks for document {doc_id} to collection {collection_name}")
            
            return {
                "success": True,
                "chunks_added": len(chunks),
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Failed to add chunks to collection: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "chunks_added": 0,
                "processing_time": time.time() - start_time
            }
    
    async def semantic_search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        threshold: float = 0.7,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform semantic search in a collection.
        
        Args:
            query: Search query
            collection_name: Collection name
            top_k: Number of results to return
            threshold: Minimum similarity score threshold
            where: Metadata filter
            
        Returns:
            List[Dict]: Search results
        """
        try:
            # Get the collection
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            
            # Perform the search
            results = collection.query(
                query_texts=[query],
                n_results=top_k * 2,  # Get more results than needed for threshold filtering
                where=where
            )
            
            # Process results
            processed_results = []
            
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity score (ChromaDB returns L2 distance)
                # Lower distance = higher similarity
                similarity = 1.0 - min(distance, 1.0)
                
                # Apply threshold
                if similarity >= threshold:
                    processed_results.append({
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata,
                        "similarity": similarity
                    })
            
            # Sort by similarity and limit to top_k
            processed_results.sort(key=lambda x: x["similarity"], reverse=True)
            processed_results = processed_results[:top_k]
            
            # Log search analytics
            self._log_search_analytics(query, collection_name, len(processed_results), where)
            
            return processed_results
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []
    
    async def hybrid_search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform hybrid search (semantic + keyword) in a collection.
        
        Args:
            query: Search query
            collection_name: Collection name
            top_k: Number of results to return
            semantic_weight: Weight of semantic search vs keyword search
            where: Metadata filter
            
        Returns:
            List[Dict]: Search results
        """
        try:
            # Get the collection
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            
            # Perform semantic search
            semantic_results = collection.query(
                query_texts=[query],
                n_results=top_k * 2,
                where=where
            )
            
            # Perform keyword search
            # ChromaDB doesn't have built-in keyword search, so we'll get all documents
            # and filter them manually
            all_docs = collection.get(where=where)
            
            # Simple keyword matching
            keyword_matches = []
            query_terms = set(query.lower().split())
            
            for i, (doc_id, document, metadata) in enumerate(zip(
                all_docs["ids"],
                all_docs["documents"],
                all_docs["metadatas"]
            )):
                # Count matching terms
                doc_terms = set(document.lower().split())
                matching_terms = query_terms.intersection(doc_terms)
                
                if matching_terms:
                    # Calculate a simple TF score
                    score = len(matching_terms) / len(query_terms)
                    keyword_matches.append({
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata,
                        "keyword_score": score
                    })
            
            # Sort keyword matches by score
            keyword_matches.sort(key=lambda x: x["keyword_score"], reverse=True)
            keyword_matches = keyword_matches[:top_k * 2]
            
            # Combine results
            combined_results = {}
            
            # Add semantic results
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                semantic_results["ids"][0],
                semantic_results["documents"][0],
                semantic_results["metadatas"][0],
                semantic_results["distances"][0]
            )):
                similarity = 1.0 - min(distance, 1.0)
                combined_results[doc_id] = {
                    "id": doc_id,
                    "content": document,
                    "metadata": metadata,
                    "semantic_score": similarity,
                    "keyword_score": 0.0
                }
            
            # Add keyword results
            for match in keyword_matches:
                doc_id = match["id"]
                if doc_id in combined_results:
                    combined_results[doc_id]["keyword_score"] = match["keyword_score"]
                else:
                    combined_results[doc_id] = {
                        "id": doc_id,
                        "content": match["content"],
                        "metadata": match["metadata"],
                        "semantic_score": 0.0,
                        "keyword_score": match["keyword_score"]
                    }
            
            # Calculate combined scores
            results = list(combined_results.values())
            for result in results:
                result["score"] = (
                    semantic_weight * result["semantic_score"] +
                    (1 - semantic_weight) * result["keyword_score"]
                )
            
            # Sort by combined score and limit to top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]
            
            # Log search analytics
            self._log_search_analytics(query, collection_name, len(results), where, hybrid=True)
            
            return results
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            return []
    
    def _log_search_analytics(
        self,
        query: str,
        collection_name: str,
        result_count: int,
        where: Optional[Dict] = None,
        hybrid: bool = False
    ) -> None:
        """
        Log search analytics.
        
        Args:
            query: Search query
            collection_name: Collection name
            result_count: Number of results
            where: Metadata filter
            hybrid: Whether this was a hybrid search
        """
        try:
            # Create analytics entry
            analytics = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "collection": collection_name,
                "result_count": result_count,
                "search_type": "hybrid" if hybrid else "semantic",
                "filters": where
            }
            
            # In a real implementation, you might store this in a database
            # For now, we'll just log it
            logger.info(f"Search analytics: {json.dumps(analytics)}")
        except Exception as e:
            logger.error(f"Failed to log search analytics: {str(e)}")
    
    async def load_from_huggingface(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        split: str = "train",
        text_column: str = "text",
        collection_name: Optional[str] = None,
        chunk_strategy: str = "token",
        max_documents: Optional[int] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict:
        """
        Load documents from a Hugging Face dataset.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face
            subset: Subset of the dataset (optional)
            split: Split of the dataset (default: "train")
            text_column: Column containing the text data
            collection_name: Name of the collection to store documents
            chunk_strategy: Chunking strategy
            max_documents: Maximum number of documents to process
            progress_callback: Callback for progress updates
            
        Returns:
            Dict: Processing results
        """
        start_time = time.time()
        try:
            # Fix for PyArrow compatibility issue
            import pyarrow
            if not hasattr(pyarrow, 'PyExtensionType') and hasattr(pyarrow, 'ExtensionType'):
                pyarrow.PyExtensionType = pyarrow.ExtensionType
                
            import datasets
            
            # Load the dataset
            if progress_callback:
                progress_callback(0.1)
                
            if subset:
                dataset = datasets.load_dataset(dataset_name, subset, split=split)
            else:
                dataset = datasets.load_dataset(dataset_name, split=split)
            
            # Limit documents if specified
            if max_documents and max_documents < len(dataset):
                dataset = dataset.select(range(max_documents))
            
            if progress_callback:
                progress_callback(0.2)
            
            # Use dataset name as collection name if not provided
            if not collection_name:
                collection_name = dataset_name.split('/')[-1]
                
            # Create collection if it doesn't exist
            try:
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
            except Exception:
                await self.create_collection(collection_name)
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
            
            # Process each document in the dataset
            total_chunks = 0
            processed_docs = 0
            failed_docs = 0
            total_docs = len(dataset)
            
            for i, item in enumerate(dataset):
                if text_column not in item or not item[text_column]:
                    logger.warning(f"Text column '{text_column}' not found or empty in item {i}")
                    failed_docs += 1
                    continue
                    
                text = item[text_column]
                
                # Generate a document ID
                doc_id = f"huggingface_{dataset_name.replace('/', '_')}_{i}"
                
                # Create metadata from other columns
                metadata = {k: v for k, v in item.items() 
                           if k != text_column and isinstance(v, (str, int, float, bool))}
                metadata["source"] = f"huggingface/{dataset_name}"
                
                # Chunk the text
                chunks = self._chunk_text(text, chunk_strategy)
                
                # Prepare document chunks
                chunk_ids = []
                chunk_texts = []
                chunk_metadatas = []
                
                # Process chunks
                for j, chunk in enumerate(chunks):
                    # Generate chunk ID
                    chunk_id = self._generate_chunk_id(chunk, doc_id, j)
                    
                    # Create chunk metadata
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "doc_id": doc_id,
                        "chunk_index": j,
                        "chunk_total": len(chunks),
                        "timestamp": datetime.now().isoformat(),
                        "chunk_strategy": chunk_strategy
                    })
                    
                    # Add to lists
                    chunk_ids.append(chunk_id)
                    chunk_texts.append(chunk)
                    chunk_metadatas.append(chunk_metadata)
                
                # Add chunks to collection
                if chunk_ids:  # Only add if we have chunks
                    try:
                        # Check for existing chunks with the same doc_id
                        existing_chunks = collection.get(where={"doc_id": doc_id})
                        
                        # Delete existing chunks if found
                        if existing_chunks["ids"]:
                            collection.delete(ids=existing_chunks["ids"])
                        
                        # Add new chunks
                        collection.add(
                            ids=chunk_ids,
                            documents=chunk_texts,
                            metadatas=chunk_metadatas
                        )
                        
                        total_chunks += len(chunks)
                        processed_docs += 1
                    except Exception as e:
                        logger.error(f"Failed to add chunks for document {doc_id}: {str(e)}")
                        failed_docs += 1
                
                # Update progress
                if progress_callback:
                    progress_callback(0.2 + 0.8 * (i / total_docs))
            
            logger.info(f"Loaded {processed_docs} documents with {total_chunks} chunks from {dataset_name}")
            
            return {
                "success": True,
                "dataset": dataset_name,
                "documents_processed": processed_docs,
                "documents_failed": failed_docs,
                "chunks_added": total_chunks,
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Failed to load from Hugging Face: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            
    async def delete_document(self, doc_id: str, collection_name: str) -> bool:
        """
        Delete a document from a collection.
        
        Args:
            doc_id: Document ID
            collection_name: Collection name
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get the collection
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            
            # Find chunks for this document
            results = collection.get(
                where={"doc_id": doc_id}
            )
            
            if not results["ids"]:
                logger.warning(f"No chunks found for document {doc_id} in collection {collection_name}")
                return False
            
            # Delete the chunks
            collection.delete(
                ids=results["ids"]
            )
            
            logger.info(f"Deleted {len(results['ids'])} chunks for document {doc_id} from collection {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {str(e)}")
            return False

# Example usage
async def example_usage():
    # Initialize vector store
    vector_store = VectorStore(host="localhost", port=8000)
    
    # List collections
    collections = await vector_store.list_collections()
    print(f"Available collections: {collections}")
    
    # Create a collection
    collection_name = "huggingface_collection"
    success = await vector_store.create_collection(
        name=collection_name,
        metadata={"description": "Hugging Face dataset collection"}
    )
    print(f"Created collection: {success}")
    
    # Load data from Hugging Face dataset
    result = await vector_store.load_from_huggingface(
        dataset_name="prathyushreddy1991/legalMVP",  # Legal dataset
        text_column="text",       # Column containing document text
        collection_name=collection_name,
        max_documents=100         # Limit for testing
    )
    print(f"Hugging Face dataset loading result: {result}")
    
    if result["success"]:
        # Perform semantic search
        search_results = await vector_store.semantic_search(
            query="What are the key provisions of the Americans with Disabilities Act?",
            collection_name=collection_name,
            top_k=3
        )
        print(f"Semantic search results: {search_results}")
        
        # Perform hybrid search
        hybrid_results = await vector_store.hybrid_search(
            query="What are the key provisions of the Americans with Disabilities Act?",
            collection_name=collection_name,
            top_k=3
        )
        print(f"Hybrid search results: {hybrid_results}")
    
    # Backup the collection
    backup_path = await vector_store.backup_collection(collection_name)
    print(f"Backed up collection to: {backup_path}")
    
    # Delete the collection
    delete_success = await vector_store.delete_collection(collection_name)
    print(f"Deleted collection: {delete_success}")
    
    # Restore the collection if backup was successful
    if backup_path:
        restore_success = await vector_store.restore_collection(backup_path)
        print(f"Restored collection: {restore_success}")

if __name__ == "__main__":
    # Run the example with Hugging Face dataset
    asyncio.run(example_usage())
