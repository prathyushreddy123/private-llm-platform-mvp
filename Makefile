# Makefile for Private LLM Platform

# Variables
DOCKER_COMPOSE = docker compose -f docker-compose.yml
DOCKER_COMPOSE_DEV = docker compose -f docker-compose.yml -f docker-compose.override.yml
DOCKER_COMPOSE_PROD = docker compose -f docker-compose.yml -f docker-compose.prod.yml
ENV_FILE = docker/.env

# Default target
.PHONY: help
help:
	@echo "Private LLM Platform Makefile"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Development targets:"
	@echo "  up              Start all services in development mode"
	@echo "  down            Stop all services"
	@echo "  restart         Restart all services"
	@echo "  logs            View logs from all services"
	@echo "  logs-SERVICE    View logs from a specific service (e.g., logs-ollama)"
	@echo "  shell-SERVICE   Open a shell in a specific service (e.g., shell-backend)"
	@echo "  ps              Show service status"
	@echo ""
	@echo "Production targets:"
	@echo "  prod-up         Start all services in production mode"
	@echo "  prod-down       Stop all production services"
	@echo "  prod-restart    Restart all production services"
	@echo "  prod-logs       View logs from all production services"
	@echo ""
	@echo "Data management:"
	@echo "  backup          Create a backup of all data"
	@echo "  restore         Restore data from backup"
	@echo "  clean           Remove all data volumes"
	@echo ""
	@echo "Maintenance:"
	@echo "  build           Build all images"
	@echo "  pull            Pull latest images"
	@echo "  prune           Remove unused Docker resources"
	@echo "  update-models   Update Ollama models"
	@echo ""
	@echo "Setup:"
	@echo "  setup           Initial setup of the platform"
	@echo "  create-env      Create .env file from example"

# Check if .env file exists
check-env:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "Error: $(ENV_FILE) not found. Run 'make create-env' first."; \
		exit 1; \
	fi

# Development targets
.PHONY: up
up: check-env
	@echo "Starting services in development mode..."
	@cd docker && $(DOCKER_COMPOSE) up -d
	@echo "Services are starting. View logs with 'make logs'"

.PHONY: down
down:
	@echo "Stopping services..."
	@cd docker && $(DOCKER_COMPOSE) down
	@echo "Services stopped"

.PHONY: restart
restart: down up

.PHONY: logs
logs:
	@cd docker && $(DOCKER_COMPOSE) logs -f

.PHONY: logs-%
logs-%:
	@cd docker && $(DOCKER_COMPOSE) logs -f $*

.PHONY: shell-%
shell-%:
	@cd docker && $(DOCKER_COMPOSE) exec $* sh || $(DOCKER_COMPOSE) exec $* bash

.PHONY: ps
ps:
	@cd docker && $(DOCKER_COMPOSE) ps

# Production targets
.PHONY: prod-up
prod-up: check-env
	@echo "Starting services in production mode..."
	@cd docker && $(DOCKER_COMPOSE_PROD) up -d
	@echo "Production services are starting. View logs with 'make prod-logs'"

.PHONY: prod-down
prod-down:
	@echo "Stopping production services..."
	@cd docker && $(DOCKER_COMPOSE_PROD) down
	@echo "Production services stopped"

.PHONY: prod-restart
prod-restart: prod-down prod-up

.PHONY: prod-logs
prod-logs:
	@cd docker && $(DOCKER_COMPOSE_PROD) logs -f

# Data management
.PHONY: backup
backup: check-env
	@echo "Creating backup..."
	@mkdir -p docker/backups
	@BACKUP_FILE="backup_$$(date +%Y%m%d_%H%M%S).tar.gz"; \
	cd docker && $(DOCKER_COMPOSE) exec postgres pg_dumpall -c -U $$(grep POSTGRES_USER .env | cut -d '=' -f2) > backups/postgres_dump.sql && \
	tar -czf backups/$$BACKUP_FILE -C volumes . && \
	echo "Backup created: docker/backups/$$BACKUP_FILE"

# New: Initialize DBs
.PHONY: init-db
init-db: check-env
	@echo "Initializing databases..."
	@cd docker && $(DOCKER_COMPOSE) exec -T postgres \
		psql -U $$(grep POSTGRES_USER .env | cut -d '=' -f2) -d postgres -v ON_ERROR_STOP=1 -c "\
		DO $$ \
		BEGIN \
		    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'llm_n8n') THEN \
		        CREATE DATABASE llm_n8n; \
		    END IF; \
		END \
		$$;"
	@echo "Database llm_n8n ensured."

.PHONY: restore
restore: check-env
	@echo "Restoring from backup..."
	@echo "Available backups:"
	@ls -1 docker/backups/*.tar.gz 2>/dev/null || echo "No backups found"
	@read -p "Enter backup file name (or press enter to cancel): " BACKUP_FILE; \
	if [ -z "$$BACKUP_FILE" ]; then \
		echo "Restore cancelled"; \
		exit 0; \
	fi; \
	if [ ! -f "docker/backups/$$BACKUP_FILE" ]; then \
		echo "Error: Backup file not found"; \
		exit 1; \
	fi; \
	cd docker && $(DOCKER_COMPOSE) down && \
	echo "Restoring data..." && \
	rm -rf volumes/* && \
	tar -xzf backups/$$BACKUP_FILE -C volumes && \
	$(DOCKER_COMPOSE) up -d postgres && \
	sleep 5 && \
	$(DOCKER_COMPOSE) exec -T postgres psql -U $$(grep POSTGRES_USER .env | cut -d '=' -f2) < backups/postgres_dump.sql && \
	$(DOCKER_COMPOSE) down && \
	echo "Restore completed. Start services with 'make up'"

.PHONY: clean
clean:
	@echo "WARNING: This will remove all data volumes!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		cd docker && $(DOCKER_COMPOSE) down -v; \
		echo "All volumes removed"; \
	else \
		echo "Operation cancelled"; \
	fi

# Maintenance
.PHONY: build
build: check-env
	@echo "Building images..."
	@cd docker && $(DOCKER_COMPOSE) build

.PHONY: pull
pull:
	@echo "Pulling latest images..."
	@cd docker && $(DOCKER_COMPOSE) pull

.PHONY: prune
prune:
	@echo "Removing unused Docker resources..."
	@docker system prune -f

.PHONY: update-models
update-models: check-env
	@echo "Updating Ollama models..."
	@cd docker && $(DOCKER_COMPOSE) exec ollama ollama pull $$(grep OLLAMA_MODELS .env | cut -d '=' -f2)

# Setup
.PHONY: setup
setup: create-env
	@echo "Setting up Private LLM Platform..."
	@mkdir -p docker/volumes/{postgres/{data,backup},chromadb,ollama,redis,n8n,backend_temp}
	@chmod +x scripts/*.sh
	@echo "Setup complete. Start the platform with 'make up'"

.PHONY: create-env
create-env:
	@if [ -f $(ENV_FILE) ]; then \
		echo "$(ENV_FILE) already exists. Skipping."; \
	else \
		echo "Creating $(ENV_FILE) from example..."; \
		cp docker/.env.example $(ENV_FILE); \
		echo "$(ENV_FILE) created. Please edit it with your configuration."; \
	fi

