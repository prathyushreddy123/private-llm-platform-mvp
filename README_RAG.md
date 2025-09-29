# Production-Ready RAG Pipeline

This document describes the Retrieval-Augmented Generation (RAG) pipeline implemented in this project. The RAG pipeline combines the power of vector databases for retrieval with language models for generation, providing accurate and contextually relevant responses to user queries.

## Architecture

The RAG pipeline consists of the following components:

1. **Vector Store**: ChromaDB-based vector storage for document embeddings
2. **RAG Pipeline**: Core logic for retrieval, context processing, and generation
3. **API Layer**: FastAPI endpoints for interacting with the RAG pipeline
4. **UI Layer**: Streamlit interface for user interaction

### Key Features

- **Multi-stage retrieval**: Coarse to fine-grained retrieval process
- **Context optimization**: Dynamic context window sizing based on relevance
- **Query expansion**: Improves retrieval by generating multiple query variations
- **Response validation**: Checks for hallucinations and relevance
- **Source attribution**: Links generated content back to source documents
- **Confidence scoring**: Provides confidence metrics for generated responses

## Components

### Vector Store (`vector_store.py`)

The Vector Store module provides:
- Collection management (create, delete, backup, restore)
- Document processing and chunking
- Embedding generation and storage
- Semantic and hybrid search capabilities

### RAG Pipeline (`rag_pipeline.py`)

The RAG Pipeline module implements:
- Multi-stage retrieval with re-ranking
- Context optimization and selection
- Prompt construction with domain-specific templates
- LLM integration for response generation
- Response validation and quality control
- Citation extraction and source attribution

### API Layer (`rag_api.py`)

The API Layer provides:
- RESTful endpoints for querying the RAG pipeline
- Collection management endpoints
- Health check and monitoring endpoints
- Configuration options for retrieval and generation

### UI Layer (`frontend/pages/rag_chat.py`)

The UI Layer offers:
- Chat interface for interacting with the RAG pipeline
- Collection and domain selection
- Advanced settings for retrieval and generation parameters
- Context and citation visualization
- Confidence and performance metrics

## Usage

### Running the RAG Pipeline

1. Start the ChromaDB service:
   ```
   docker-compose up -d chroma
   ```

2. Start the Ollama service:
   ```
   docker-compose up -d ollama
   ```

3. Start the API server:
   ```
   cd backend
   uvicorn api.rag_api:app --host 0.0.0.0 --port 8080 --reload
   ```

4. Start the Streamlit UI:
   ```
   cd frontend
   streamlit run app.py
   ```

### Example Usage

```python
from api.vector_store import VectorStore
from api.rag_pipeline import RAGPipeline

# Initialize components
vector_store = VectorStore(host="localhost", port=8000)
rag = RAGPipeline(vector_store=vector_store, llm_service="http://localhost:11434")

# Process a query
result = await rag.process_query(
    query="What are the key provisions of the Americans with Disabilities Act?",
    collection="legal_documents",
    domain="legal"
)

# Display the result
print(f"Response: {result['response']}")
print(f"Confidence: {result['confidence_score']}")
```

## Configuration

### Retrieval Configuration

- `chunk_size`: Size of text chunks for processing
- `overlap`: Overlap between chunks
- `top_k`: Number of results to retrieve
- `threshold`: Minimum similarity threshold
- `rerank_threshold`: Threshold for re-ranking
- `max_tokens_limit`: Maximum tokens for context

### Generation Configuration

- `temperature`: Controls randomness (0.0 to 1.0)
- `top_p`: Controls diversity (0.0 to 1.0)
- `max_tokens`: Maximum tokens to generate
- `model`: LLM model to use
- `system_prompt`: System prompt for the LLM

## Quality Control

The RAG pipeline implements several quality control mechanisms:

1. **Confidence Scoring**: Each response includes a confidence score based on:
   - Source support (how well the response is grounded in the retrieved context)
   - Hallucination detection (identifying unsupported claims)
   - Query relevance (how well the response addresses the query)

2. **Response Validation**: Validates responses for:
   - Factual accuracy against retrieved sources
   - Response length and completeness
   - Relevance to the original query

3. **Citation Extraction**: Links response claims back to source documents

## Evaluation

The RAG pipeline can be evaluated using:

1. **RAGAS Metrics**:
   - Faithfulness (absence of hallucinations)
   - Answer relevance (relevance to query)
   - Context relevance (relevance of retrieved context)
   - Context recall (recall of relevant information)

2. **Custom Metrics**:
   - Response time
   - Confidence score distribution
   - Source utilization (how many sources contributed to the response)

## Future Improvements

1. **Enhanced Retrieval**:
   - Implement hybrid dense-sparse retrieval
   - Add support for structured data retrieval
   - Implement cross-encoder re-ranking

2. **Advanced Generation**:
   - Implement chain-of-thought reasoning
   - Add support for multi-step generation
   - Implement consensus generation across multiple models

3. **Quality Control**:
   - Improve hallucination detection with external fact verification
   - Add support for uncertainty quantification
   - Implement automatic evaluation with RAGAS

4. **Performance Optimization**:
   - Add caching for frequent queries
   - Implement batch processing for document ingestion
   - Optimize context selection for different LLM context windows
