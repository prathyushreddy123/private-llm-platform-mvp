# RAG Implementation: Detailed Process Flow

## Overview

This document explains the Retrieval-Augmented Generation (RAG) implementation in this project, detailing how data flows through the system, the interaction with ChromaDB, and the step-by-step process from query to response.

## Architecture Components

- **Vector Store (`vector_store.py`)**: Manages document storage, embedding, and retrieval
- **RAG Pipeline (`rag_pipeline.py`)**: Orchestrates the entire RAG process
- **Ollama Service (`ollama_service.py`)**: Handles interaction with the LLM
- **RAG API (`rag_api.py`)**: Provides API endpoints for the RAG system
- **ChromaDB**: Vector database for storing and retrieving document embeddings
- **Ollama**: Local LLM service for generating responses

## Detailed Process Flow

### 1. Document Ingestion

1. **Document Loading**:
   - Documents are loaded from various sources (files, Hugging Face datasets)
   - `VectorStore.load_from_huggingface()` or `VectorStore.load_from_files()` methods handle this process

2. **Document Chunking**:
   - Documents are split into smaller chunks for better retrieval
   - Two chunking strategies are available:
     - Token-based splitting (`TokenTextSplitter`)
     - Character-based recursive splitting (`RecursiveCharacterTextSplitter`)

3. **Embedding Generation**:
   - Each chunk is converted to a vector embedding using SentenceTransformer
   - Default model is "all-MiniLM-L6-v2"
   - Custom embedding function handles the embedding process:
   ```python
   def embed_func(texts):
       embeddings = model.encode(texts)
       return embeddings.tolist()
   ```

4. **Storage in ChromaDB**:
   - Chunks and their embeddings are stored in ChromaDB collections
   - Each chunk includes metadata like source, timestamp, and chunk position
   - Collections can be backed up, restored, and managed through the VectorStore class

### 2. Query Processing

1. **Query Reception**:
   - User query is received through the RAG API or example script
   - `RAGPipeline.process_query()` initiates the processing

2. **Query Expansion**:
   - Original query is expanded into multiple variations to improve retrieval coverage
   - Variations include specific information requests, definitions, and contextual queries
   ```python
   # Example expansion:
   expanded_queries = [
       original_query,
       f"specific information about {original_query}",
       f"definition of {original_query.lower()}"
   ]
   ```

### 3. Retrieval Process

1. **Multi-Stage Retrieval**:
   - For each expanded query, the system performs retrieval from ChromaDB
   - `vector_store.hybrid_search()` combines semantic and keyword search:
     ```python
     results = collection.query(
         query_texts=[query],
         n_results=top_k * 2,
         where=where
     )
     ```

2. **Result Processing**:
   - Search results are combined from all expanded queries
   - Duplicate chunks are removed
   - Each chunk receives a combined score based on semantic similarity and keyword matching

3. **Re-Ranking**:
   - Chunks are re-ranked based on multiple factors:
     - Semantic similarity (vector distance)
     - Keyword overlap between query and chunk
     - Position of query terms in the chunk
   ```python
   relevance_score = (
       0.6 * semantic_score +
       0.3 * overlap +
       0.1 * position_score
   )
   ```

4. **Context Optimization**:
   - Top chunks are selected based on relevance score
   - Context window is optimized to fit within LLM token limits
   - Chunks are formatted with source information for citation

### 4. LLM Generation

1. **Prompt Construction**:
   - A domain-specific prompt template is selected (default, legal, medical, technical)
   - The template is filled with:
     - The query
     - Optimized context from retrieved chunks
     - System instructions for response format
   ```python
   prompt = self.prompt_templates[template].format(
       context=formatted_context,
       query=query
   )
   ```

2. **Ollama API Call**:
   - The constructed prompt is sent to Ollama API
   - Default model is "llama2:7b" (configurable)
   - Request includes parameters like temperature and max tokens
   ```python
   ollama_payload = {
       "model": self.generation_config["model"],
       "messages": messages,
       "stream": False,
       "options": {
           "temperature": self.generation_config.get("temperature", 0.7),
           "top_p": self.generation_config.get("top_p", 0.9),
           "top_k": self.generation_config.get("top_k", 40),
           "num_ctx": self.generation_config.get("max_tokens", 2000)
       }
   }
   ```

3. **Response Parsing**:
   - LLM response is parsed from JSON
   - Error handling manages timeouts and parsing issues

### 5. Response Validation and Enhancement

1. **Response Validation**:
   - Generated response is validated for:
     - Relevance to the query
     - Factual accuracy based on retrieved context
     - Completeness of answer

2. **Citation Extraction**:
   - Citations in the response are linked back to source documents
   - Format: `[Source X]` where X is the source index

3. **Confidence Scoring**:
   - A confidence score is calculated based on:
     - Relevance of retrieved chunks
     - LLM's stated confidence
     - Citation coverage

### 6. Result Return

1. **Response Formatting**:
   - Final response is formatted with:
     - Generated answer
     - Confidence score
     - Processing time
     - Source citations

2. **Analytics Logging**:
   - Detailed analytics are logged for each query:
     ```python
     logger.info(f"RAG analytics: {json.dumps({
         'timestamp': datetime.now().isoformat(),
         'query': query,
         'collection': collection,
         'response_length': len(response.get('response', '')),
         'context_chunks': len(context_chunks),
         'confidence_score': confidence,
         'processing_time': time.time() - start_time
     })}")
     ```

## Integration Points

### ChromaDB Integration

- **Connection**: System connects to ChromaDB at startup (local or remote)
- **Collections**: Documents are organized in named collections
- **Queries**: Vector similarity search is performed through ChromaDB's query API
- **Persistence**: Collections can be backed up and restored

### Ollama Integration

- **API Communication**: System communicates with Ollama via HTTP API
- **Model Selection**: Configurable model selection (default: llama2:7b)
- **Parameter Tuning**: Generation parameters can be adjusted
- **Error Handling**: Robust error handling for API failures

## Performance Considerations

- **Chunking Strategy**: Affects retrieval quality and processing speed
- **Embedding Model**: Determines semantic understanding quality
- **Re-ranking Algorithm**: Critical for ensuring most relevant context
- **LLM Model Choice**: Balances quality vs. speed
- **Context Window Size**: Must be optimized for LLM limits

## Usage Examples

See `rag_example.py` for a complete example of the RAG pipeline in action.