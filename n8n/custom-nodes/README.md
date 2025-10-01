# Custom n8n Nodes for LLM Operations

This directory contains custom n8n nodes designed to enhance the LLM platform with specialized functionality for RAG pipelines and document processing.

## Available Nodes

### 1. OllamaChat

A streamlined interface for interacting with Ollama models through n8n workflows.

**Features:**
- Dropdown for model selection
- Streaming support (toggle on/off)
- Adjustable parameters (temperature, top_p, top_k)
- System prompt input field
- Context window management
- Token counting utility
- Optional response caching

### 2. VectorStore

Manage documents in ChromaDB vector database directly from n8n workflows.

**Features:**
- Insert new documents
- Perform similarity search
- Update document metadata
- Delete stored documents
- Full collection management

### 3. DocumentChunker

Intelligent document preprocessing and chunking for optimal RAG performance.

**Features:**
- Multiple chunking strategies (tokens, sentences, paragraphs)
- Configurable overlap size
- Token counting support
- Metadata retention (source, page, timestamp)

## Installation

1. Navigate to the n8n custom nodes directory:
```bash
cd n8n/custom-nodes/
```

2. Install dependencies and build the nodes:
```bash
npm install
npm run build
```

3. Link the custom nodes to your n8n installation:
```bash
npm link
cd ~/.n8n
npm link n8n-nodes-ollama
```

4. Restart your n8n instance to load the custom nodes.

## Usage Examples

Example workflows demonstrating these custom nodes can be found in the `n8n/workflows/` directory:

- `document_ingestion.json`: Shows how to use DocumentChunker and VectorStore nodes to process and store documents
- `query_processing.json`: Demonstrates using VectorStore for retrieval and OllamaChat for generation
- `batch_processing.json`: Example of batch processing documents using these custom nodes

## Development

To extend or modify these nodes:

1. Update the node files in their respective directories
2. Run `npm run build` to compile changes
3. Restart n8n to apply changes

### Node Structure

Each node follows this basic structure:

```typescript
export class NodeName implements INodeType {
  description: INodeTypeDescription = {
    // Node metadata and UI configuration
  };
  
  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    // Implementation logic
  }
}
```

## Integration with RAG Pipeline

These custom nodes are designed to integrate seamlessly with the RAG pipeline:

1. **Document Processing Flow:**
   - Input documents → DocumentChunker → VectorStore
   
2. **Query Processing Flow:**
   - User query → VectorStore (retrieval) → OllamaChat (generation)

This integration enables fully automated document processing and query workflows within n8n.
