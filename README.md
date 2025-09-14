# Private LLM Platform

A production-ready, self-hosted LLM platform for enterprises in regulated industries. This system enables organizations to deploy a ChatGPT-like experience that operates entirely within their infrastructure, ensuring complete data privacy.

## Features

- **Document Processing**: Automatically ingest and index PDFs, DOCX, and TXT files
- **Intelligent Q&A**: Answer questions using uploaded documents as context
- **Privacy First**: All data stays local, with PII detection and masking
- **Enterprise Integration**: Connect to Google Drive, Slack, and email
- **Audit Trail**: Log all queries and responses for compliance

## Architecture

- **LLM Models**: Open-source models (LLaMA, Mistral, Mixtral) running locally via Ollama
- **Vector Database**: ChromaDB for semantic search and document retrieval
- **RAG Pipeline**: Retrieve relevant document chunks, then generate answers
- **Workflow Engine**: n8n for automating document processing and integrations
- **Frontend**: Streamlit for MVP
- **Infrastructure**: Docker Compose for easy deployment

## Quick Start

1. Clone this repository
2. Copy `.env.example` to `.env` and update the values
3. Run `./scripts/setup.sh` to initialize the environment
4. Start the platform with `./scripts/start.sh`
5. Open your browser to `http://localhost:8501` to access the UI

## Requirements

- Docker and Docker Compose
- 16GB RAM minimum (32GB recommended)
- 100GB disk space
- NVIDIA GPU recommended for optimal performance

## Documentation

- [Setup Guide](docs/SETUP.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [API Documentation](docs/API.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

