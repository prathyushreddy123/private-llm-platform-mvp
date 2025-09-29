"""
FastAPI endpoints for the RAG pipeline.

This module provides REST API endpoints for querying the RAG pipeline,
managing collections, and monitoring performance.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import RAG components
from .vector_store import VectorStore
from .rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_api")

# Initialize FastAPI app
app = FastAPI(
    title="Legal RAG API",
    description="API for Retrieval-Augmented Generation on legal documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
vector_store = None
rag_pipeline = None

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., description="The user query")
    collection: str = Field(..., description="The collection to search in")
    domain: str = Field("legal", description="Domain for prompt template selection")
    chat_history: List[Dict[str, str]] = Field(default_factory=list, description="Previous conversation history")
    retrieval_options: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Optional retrieval configuration parameters"
    )
    generation_options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional generation configuration parameters"
    )

class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    query: str = Field(..., description="The original query")
    response: str = Field(..., description="The generated response")
    confidence_score: float = Field(..., description="Confidence score for the response")
    processing_time: float = Field(..., description="Processing time in seconds")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Source citations")
    context_metadata: List[Dict[str, Any]] = Field(default_factory=list, description="Metadata for context chunks")

class CollectionInfo(BaseModel):
    """Model for collection information."""
    name: str = Field(..., description="Collection name")
    count: int = Field(..., description="Number of documents in collection")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Collection metadata")

class CollectionList(BaseModel):
    """Model for listing collections."""
    collections: List[str] = Field(..., description="List of collection names")

class StatusResponse(BaseModel):
    """Model for status responses."""
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")

# Dependency for initializing components
async def get_vector_store():
    """Initialize and return the vector store."""
    global vector_store
    if vector_store is None:
        host = os.environ.get("CHROMA_HOST", "localhost")
        port = int(os.environ.get("CHROMA_PORT", 8000))
        vector_store = VectorStore(host=host, port=port)
        logger.info(f"Initialized vector store with host={host}, port={port}")
    return vector_store

async def get_rag_pipeline(vector_store: VectorStore = Depends(get_vector_store)):
    """Initialize and return the RAG pipeline."""
    global rag_pipeline
    if rag_pipeline is None:
        llm_service = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
        model = os.environ.get("DEFAULT_LLM_MODEL", "llama2")
        rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            llm_service=llm_service,
            config={
                "model": model,
                "temperature": 0.7
            }
        )
        logger.info(f"Initialized RAG pipeline with LLM service at {llm_service}")
    return rag_pipeline

# API endpoints
@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint providing API information."""
    return {
        "status": "success",
        "message": "Legal RAG API is running",
        "details": {
            "version": "1.0.0",
            "documentation": "/docs"
        }
    }

@app.get("/health", response_model=StatusResponse)
async def health_check(
    vector_store: VectorStore = Depends(get_vector_store),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Health check endpoint."""
    try:
        # Check vector store connection
        collections = await vector_store.list_collections()
        
        return {
            "status": "success",
            "message": "All systems operational",
            "details": {
                "vector_store": "connected",
                "collections_count": len(collections),
                "rag_pipeline": "initialized"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"System health check failed: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Process a query through the RAG pipeline."""
    try:
        # Update retrieval and generation options if provided
        if request.retrieval_options:
            rag_pipeline.retrieval_config.update(request.retrieval_options)
        
        if request.generation_options:
            rag_pipeline.generation_config.update(request.generation_options)
        
        # Process the query
        result = await rag_pipeline.process_query(
            query=request.query,
            collection=request.collection,
            chat_history=request.chat_history,
            domain=request.domain
        )
        
        # Format the response
        response = {
            "query": request.query,
            "response": result["response"],
            "confidence_score": result["confidence_score"],
            "processing_time": result["processing_time"],
            "citations": result["citations"],
            "context_metadata": result["context_metadata"]
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )

@app.get("/collections", response_model=CollectionList)
async def list_collections(
    vector_store: VectorStore = Depends(get_vector_store)
):
    """List all available collections."""
    try:
        collections = await vector_store.list_collections()
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Failed to list collections: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list collections: {str(e)}"
        )

@app.get("/collections/{collection_name}", response_model=CollectionInfo)
async def get_collection_info(
    collection_name: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get information about a specific collection."""
    try:
        info = await vector_store.get_collection_info(collection_name)
        if not info:
            raise HTTPException(
                status_code=404,
                detail=f"Collection {collection_name} not found"
            )
        return info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get collection info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get collection info: {str(e)}"
        )

@app.post("/collections/{collection_name}", response_model=StatusResponse)
async def create_collection(
    collection_name: str,
    background_tasks: BackgroundTasks,
    metadata: Optional[Dict[str, Any]] = None,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Create a new collection."""
    try:
        success = await vector_store.create_collection(
            name=collection_name,
            metadata=metadata
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Collection {collection_name} created successfully",
                "details": {"collection_name": collection_name}
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create collection {collection_name}"
            )
    except Exception as e:
        logger.error(f"Failed to create collection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create collection: {str(e)}"
        )

@app.delete("/collections/{collection_name}", response_model=StatusResponse)
async def delete_collection(
    collection_name: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Delete a collection."""
    try:
        # First check if the collection exists
        collections = await vector_store.list_collections()
        if collection_name not in collections:
            raise HTTPException(
                status_code=404,
                detail=f"Collection {collection_name} not found"
            )
        
        # Delete the collection
        success = await vector_store.delete_collection(collection_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Collection {collection_name} deleted successfully",
                "details": {"collection_name": collection_name}
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete collection {collection_name}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete collection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete collection: {str(e)}"
        )

@app.post("/collections/{collection_name}/backup", response_model=StatusResponse)
async def backup_collection(
    collection_name: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Backup a collection to a file."""
    try:
        # Check if the collection exists
        collections = await vector_store.list_collections()
        if collection_name not in collections:
            raise HTTPException(
                status_code=404,
                detail=f"Collection {collection_name} not found"
            )
        
        # Backup the collection
        backup_path = await vector_store.backup_collection(collection_name)
        
        if backup_path:
            return {
                "status": "success",
                "message": f"Collection {collection_name} backed up successfully",
                "details": {"backup_path": backup_path}
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to backup collection {collection_name}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to backup collection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to backup collection: {str(e)}"
        )

@app.post("/collections/{collection_name}/restore", response_model=StatusResponse)
async def restore_collection(
    collection_name: str,
    backup_path: str = Query(..., description="Path to the backup file"),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Restore a collection from a backup file."""
    try:
        # Restore the collection
        success = await vector_store.restore_collection(backup_path)
        
        if success:
            return {
                "status": "success",
                "message": f"Collection restored successfully from {backup_path}",
                "details": {"collection_name": collection_name}
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to restore collection from {backup_path}"
            )
    except Exception as e:
        logger.error(f"Failed to restore collection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to restore collection: {str(e)}"
        )

# Main function to run the API with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "rag_api:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )
