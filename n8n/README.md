# n8n Workflows for Private LLM Platform

This directory contains n8n workflow templates for integrating with the Private LLM Platform. These workflows automate document processing, query handling, and batch operations.

## Workflow Overview

### 1. Document Ingestion Pipeline
- **File**: `workflows/document_ingestion.json`
- **Purpose**: Process individual document uploads and store them in the vector database
- **Trigger**: Webhook endpoint for file uploads
- **Features**:
  - File type detection and routing
  - Document parsing (PDF, DOCX, TXT)
  - Text chunking with 100-token segments and 50% overlap
  - Embedding generation
  - ChromaDB storage
  - Notifications for success/failure

### 2. Query Processing
- **File**: `workflows/query_processing.json`
- **Purpose**: Handle user queries against the vector database
- **Trigger**: Webhook endpoint for query requests
- **Features**:
  - Query validation and sanitization
  - Response caching
  - Vector similarity search
  - Context ranking and optimization
  - Intelligent prompt construction
  - LLM integration via Ollama API
  - Response validation
  - Analytics logging

### 3. Batch Document Processing
- **File**: `workflows/batch_processing.json`
- **Purpose**: Process multiple documents on a schedule
- **Trigger**: Daily schedule (configurable)
- **Features**:
  - Google Drive folder monitoring
  - Queue management
  - Parallel document processing
  - Progress tracking
  - Report generation and delivery

## Setup Instructions

### Prerequisites
1. n8n server running (already configured in docker-compose.yml)
2. Access to the Private LLM Platform API
3. Required credentials:
   - Google Drive (for batch processing)
   - SMTP (for email notifications)
   - Slack/Telegram (for notifications)

### Environment Variables
Set the following environment variables in your n8n instance:

```
# API Keys
DOCUMENT_INGESTION_API_KEY=your_api_key
QUERY_API_KEY=your_api_key
RAG_API_KEY=your_api_key

# Service URLs
RAG_API_URL=http://fastapi:8080
OLLAMA_API_URL=http://ollama:11434
CHROMA_API_URL=http://chroma:8000

# Storage
CHROMA_COLLECTION=default

# Google Drive
GOOGLE_DRIVE_FOLDER_ID=your_folder_id

# Notifications
EMAIL_FROM=notifications@example.com
EMAIL_TO=admin@example.com
SLACK_CHANNEL_ID=your_channel_id
```

### Importing Workflows

1. Access your n8n instance at http://localhost:5678
2. Go to "Workflows" in the sidebar
3. Click "Import from File"
4. Select the workflow JSON file from this directory
5. Configure credentials when prompted
6. Activate the workflow

## Error Handling

All workflows include:
- Retry logic with exponential backoff
- Error notifications
- Detailed logging
- Fallback mechanisms

## Customization

### Document Processing
- Adjust chunking parameters in the "Text Chunking" node
- Modify embedding model in the "Embedding Generation" node
- Add support for additional file types by creating new parser nodes

### Query Processing
- Customize prompt templates in the "Prompt Construction" node
- Adjust relevance scoring in the "Context Ranking" node
- Modify LLM parameters in the "LLM Call" node

### Batch Processing
- Change schedule in the "Schedule Trigger" node
- Adjust queue management logic in the "Queue Management" node
- Customize report format in the "Generate Report" node

## Monitoring

- All workflows log processing statistics
- Success/failure notifications are sent via email and Slack
- Processing reports include detailed metrics

## Security Considerations

- All API endpoints are secured with API keys
- File size limits prevent abuse
- Input validation prevents injection attacks
- Sensitive information is stored in environment variables

## Performance Optimization

- Batching is used for embedding generation
- Parallel processing for batch operations
- Caching for repeated queries
- Queue management for resource-intensive operations
