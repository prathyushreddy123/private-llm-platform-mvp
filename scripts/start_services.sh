#!/bin/bash
# Start all services
service postgresql start || echo "PostgreSQL failed to start"
service redis-server start || echo "Redis failed to start"
systemctl start ollama || echo "Ollama failed to start"
systemctl start chromadb || echo "ChromaDB failed to start"
systemctl start n8n || echo "n8n failed to start"

echo "All services started. Starting Streamlit frontend..."

# Start Streamlit frontend
cd /workspace/private-llm-platform-mvp
source venv/bin/activate
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0


