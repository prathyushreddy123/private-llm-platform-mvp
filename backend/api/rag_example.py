"""
Example script demonstrating the RAG pipeline usage.
"""

import asyncio
import logging
from backend.api.vector_store import VectorStore
from backend.api.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_example")

async def run_rag_example():
    """Run a complete RAG pipeline example."""
    
    # Step 1: Initialize the vector store
    logger.info("Initializing vector store...")
    vector_store = VectorStore(host="localhost", port=8000)
    
    # Step 2: List available collections
    collections = await vector_store.list_collections()
    logger.info(f"Available collections: {collections}")
    
    # Step 3: Create a collection if it doesn't exist
    collection_name = "huggingface_collection"
    if collection_name not in collections:
        logger.info(f"Creating collection: {collection_name}")
        await vector_store.create_collection(
            name=collection_name,
            metadata={"description": "Hugging Face dataset collection"}
        )
    
    # Step 4: Load data from Hugging Face dataset if collection is empty
    collection_info = await vector_store.get_collection_info(collection_name)
    if collection_info.get("count", 0) == 0:
        logger.info("Loading data from Hugging Face dataset...")
        result = await vector_store.load_from_huggingface(
            dataset_name="prathyushreddy1991/legalMVP",
            text_column="text",
            collection_name=collection_name,
            max_documents=100
        )
        logger.info(f"Loaded {result.get('documents_processed', 0)} documents with {result.get('chunks_added', 0)} chunks")
    else:
        logger.info(f"Collection already contains {collection_info.get('count', 0)} documents")
    
    # Step 5: Initialize the RAG pipeline
    logger.info("Initializing RAG pipeline...")
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_service="http://localhost:11434",
        config={
            "model": "llama2:7b",  # Using llama2:7b for faster processing
            "temperature": 0.7,
            "top_k": 5,
            "system_prompt": "You are a legal assistant that provides accurate information based on the given context. Answer questions clearly and concisely."
        }
    )
    
    # Step 6: Process a query
    query = "What are the key provisions of the Americans with Disabilities Act?"
    logger.info(f"Processing query: {query}")
    
    result = await rag_pipeline.process_query(
        query=query,
        collection=collection_name,
        domain="legal"
    )
    
    # Step 7: Display results
    logger.info("RAG Pipeline Results:")
    logger.info(f"Query: {query}")
    logger.info(f"Response: {result['response']}")
    logger.info(f"Confidence: {result['confidence_score']:.2f}")
    logger.info(f"Processing time: {result['processing_time']:.2f} seconds")
    
    # Display citations
    if result["citations"]:
        logger.info("Citations:")
        for citation in result["citations"]:
            logger.info(f"- {citation['text']} (Source: {citation['source_metadata'].get('source', 'Unknown')})")
    
    # Step 8: Try another query
    second_query = "What legal protections exist for employees with disabilities?"
    logger.info(f"Processing second query: {second_query}")
    
    second_result = await rag_pipeline.process_query(
        query=second_query,
        collection=collection_name,
        domain="legal",
        chat_history=[
            {"query": query, "response": result["response"]}
        ]
    )
    
    logger.info("Second RAG Pipeline Results:")
    logger.info(f"Query: {second_query}")
    logger.info(f"Response: {second_result['response']}")
    logger.info(f"Confidence: {second_result['confidence_score']:.2f}")
    logger.info(f"Processing time: {second_result['processing_time']:.2f} seconds")

if __name__ == "__main__":
    # Run the example
    asyncio.run(run_rag_example())