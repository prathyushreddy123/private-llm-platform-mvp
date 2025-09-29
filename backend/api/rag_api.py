        llm_service = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
        model = os.environ.get("DEFAULT_LLM_MODEL", "mistral:7b-instruct")
        rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            llm_service=llm_service,
            config={
                "model": model,