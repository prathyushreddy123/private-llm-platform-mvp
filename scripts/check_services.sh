#!/bin/bash
# Check status of all services

echo "=== PostgreSQL ==="
service postgresql status

echo -e "\n=== Redis ==="
service redis-server status

echo -e "\n=== Ollama ==="
systemctl status ollama

echo -e "\n=== ChromaDB ==="
systemctl status chromadb

echo -e "\n=== n8n ==="
systemctl status n8n

echo -e "\n=== Ports ==="
netstat -tulpn | grep -E '5432|6379|11434|8000|5678|8501'


