"""
RAG Pipeline for Production-Ready Retrieval-Augmented Generation.

This module implements a robust RAG pipeline connecting vector stores with LLM services,
featuring multi-stage retrieval, context optimization, and quality control mechanisms.
"""

import os
import time
import json
import logging
import asyncio
import hashlib
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime

# Import vector store
from .vector_store import VectorStore

# LLM integration
import httpx

# Evaluation
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_pipeline")

class RAGPipeline:
    """Production-ready RAG pipeline connecting vector stores with LLMs."""
    
    def __init__(
        self, 
        vector_store: VectorStore, 
        llm_service: str = os.environ.get("OLLAMA_API_URL", "http://localhost:11434"),
        config: Dict = {}
    ):
        """
        Initialize the RAG Pipeline.
        
        Args:
            vector_store: Initialized VectorStore instance
            llm_service: URL of the LLM service (Ollama API)
            config: Configuration options
        """
        self.vector_store = vector_store
        self.llm_service = llm_service
        
        # Default retrieval configuration
        self.retrieval_config = {
            "chunk_size": config.get("chunk_size", 100),
            "overlap": config.get("overlap", 50),
            "top_k": config.get("top_k", 5),
            "threshold": config.get("threshold", 0.7),
            "rerank_threshold": config.get("rerank_threshold", 0.6),
            "max_tokens_limit": config.get("max_tokens_limit", 3000)
        }
        
        # Default generation configuration
        self.generation_config = {
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.9),
            "max_tokens": config.get("max_tokens", 500),
            "model": config.get("model", "llama2"),
            "system_prompt": config.get("system_prompt", "You are a helpful assistant.")
        }
        
        # Quality control configuration
        self.quality_config = {
            "min_confidence": config.get("min_confidence", 0.7),
            "max_hallucination_score": config.get("max_hallucination_score", 0.3),
            "min_source_overlap": config.get("min_source_overlap", 0.5)
        }
        
        # Initialize HTTP client for LLM service
        self.http_client = httpx.AsyncClient(timeout=300.0)
        
        # Load prompt templates
        self.templates = self._load_prompt_templates()
        
        logger.info(f"Initialized RAG Pipeline with LLM service at {llm_service}")
    
    def _load_prompt_templates(self) -> Dict:
        """
        Load prompt templates for different domains.
        
        Returns:
            Dict: Mapping of template names to template strings
        """
        # Default templates for different domains
        return {
            "default": """
                Answer the following question based on the provided context.
                
                Context:
                {context}
                
                Question: {query}
                
                Answer:
            """,
            
            "legal": """
                You are a legal assistant analyzing documents. Use only the provided context to answer the question.
                If the information is not in the context, say "I don't have enough information to answer this question."
                
                Context:
                {context}
                
                Question: {query}
                
                Provide a concise, accurate answer with citations to the relevant parts of the context.
                
                Answer:
            """,
            
            "medical": """
                You are a medical information assistant. Answer the question based strictly on the provided context.
                Do not include any medical advice or information not directly supported by the context.
                
                Context:
                {context}
                
                Question: {query}
                
                Answer (with citations):
            """,
            
            "technical": """
                You are a technical documentation assistant. Answer the question using only the provided technical context.
                Include code examples if relevant and available in the context.
                
                Technical Context:
                {context}
                
                Question: {query}
                
                Technical Answer:
            """
        }
    
    async def process_query(
        self, 
        query: str, 
        collection: str, 
        chat_history: List[Dict] = [],
        domain: str = "default"
    ) -> Dict:
        """
        Process a query through the full RAG pipeline.
        
        Args:
            query: User query
            collection: Vector store collection name
            chat_history: Previous conversation history
            domain: Domain for prompt template selection
            
        Returns:
            Dict: Processing results including answer and metadata
        """
        start_time = time.time()
        
        # Step 1: Query rewriting and expansion
        expanded_queries = await self._expand_query(query, chat_history)
        
        # Step 2: Multi-stage retrieval
        # First stage: Coarse retrieval
        coarse_results = []
        for expanded_query in expanded_queries:
            results = await self.vector_store.hybrid_search(
                query=expanded_query,
                collection_name=collection,
                top_k=self.retrieval_config["top_k"] * 2  # Get more results for filtering
            )
            coarse_results.extend(results)
        
        # Deduplicate results
        unique_results = self._deduplicate_results(coarse_results)
        
        # Second stage: Re-ranking
        reranked_chunks = await self._rerank_chunks(unique_results, query)
        
        # Step 3: Context optimization
        optimized_context = self._optimize_context(
            reranked_chunks, 
            max_tokens=self.retrieval_config["max_tokens_limit"]
        )
        
        # Step 4: Prompt construction
        prompt = self.build_prompt(
            query=query,
            context_chunks=optimized_context,
            template=domain,
            chat_history=chat_history
        )
        
        # Step 5: LLM generation
        generation_result = await self._generate_response(prompt)
        response = generation_result.get("response", "")
        
        # Step 6: Response validation and quality control
        validation_result = self.validate_response(response, optimized_context, query)
        
        # Step 7: Extract citations and sources
        citations = self.extract_citations(response, optimized_context)
        
        # Step 8: Prepare final result
        processing_time = time.time() - start_time
        
        result = {
            "query": query,
            "response": response,
            "citations": citations,
            "context": [chunk["content"] for chunk in optimized_context],
            "context_metadata": [chunk["metadata"] for chunk in optimized_context],
            "validation": validation_result,
            "processing_time": processing_time,
            "confidence_score": validation_result["confidence_score"]
        }
        
        # Log analytics
        self._log_rag_analytics(query, collection, result)
        
        return result
    
    async def _expand_query(self, query: str, chat_history: List[Dict]) -> List[str]:
        """
        Expand and rewrite the query to improve retrieval.
        
        Args:
            query: Original user query
            chat_history: Previous conversation history
            
        Returns:
            List[str]: List of expanded/rewritten queries
        """
        # Simple expansion for now - in production, use LLM to generate variations
        expanded_queries = [query]
        
        # Add a specification query
        if len(query.split()) > 3:
            expanded_queries.append(f"specific information about {query}")
        
        # Add a definition query if it looks like a "what is" question
        if "what" in query.lower() and "is" in query.lower():
            expanded_queries.append(f"definition of {query.lower().replace('what is', '').replace('?', '').strip()}")
        
        # Add context from recent chat history
        if chat_history:
            last_exchanges = chat_history[-2:] if len(chat_history) >= 2 else chat_history
            context_terms = []
            
            for exchange in last_exchanges:
                if "query" in exchange and exchange["query"]:
                    context_terms.extend(exchange["query"].split())
                if "response" in exchange and exchange["response"]:
                    # Extract key terms from response (simplified)
                    key_terms = [word for word in exchange["response"].split() 
                                if len(word) > 5 and word.lower() not in ["about", "would", "should", "could"]]
                    context_terms.extend(key_terms[:5])  # Limit to first 5 key terms
            
            if context_terms:
                # Create a context-aware query
                context_query = f"{query} in context of {' '.join(context_terms[:10])}"
                expanded_queries.append(context_query)
        
        logger.info(f"Expanded query '{query}' to {len(expanded_queries)} variations")
        return expanded_queries
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """
        Deduplicate retrieval results based on content.
        
        Args:
            results: List of retrieval results
            
        Returns:
            List[Dict]: Deduplicated results
        """
        seen_contents = set()
        unique_results = []
        
        for result in results:
            # Create a hash of the content
            content_hash = hashlib.md5(result["content"].encode()).hexdigest()
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    async def _rerank_chunks(self, chunks: List[Dict], query: str) -> List[Dict]:
        """
        Re-rank chunks based on relevance to query.
        
        Args:
            chunks: List of retrieved chunks
            query: User query
            
        Returns:
            List[Dict]: Re-ranked chunks
        """
        # Simple re-ranking based on combined scores
        # In production, use a dedicated re-ranking model
        
        for chunk in chunks:
            # Calculate a relevance score based on:
            # 1. Semantic similarity (if available)
            # 2. Keyword overlap
            # 3. Position of query terms in the chunk
            
            semantic_score = chunk.get("similarity", 0) or chunk.get("semantic_score", 0)
            
            # Calculate keyword overlap
            query_terms = set(query.lower().split())
            chunk_terms = set(chunk["content"].lower().split())
            overlap = len(query_terms.intersection(chunk_terms)) / max(len(query_terms), 1)
            
            # Calculate position score (higher if query terms appear earlier)
            position_score = 0
            for term in query_terms:
                if term in chunk["content"].lower():
                    # Find position of first occurrence
                    pos = chunk["content"].lower().find(term)
                    # Normalize by content length
                    normalized_pos = 1 - (pos / max(len(chunk["content"]), 1))
                    position_score += normalized_pos
            
            position_score = position_score / max(len(query_terms), 1)
            
            # Combined relevance score
            relevance_score = (
                0.6 * semantic_score +
                0.3 * overlap +
                0.1 * position_score
            )
            
            chunk["relevance_score"] = relevance_score
        
        # Sort by relevance score
        reranked = sorted(chunks, key=lambda x: x["relevance_score"], reverse=True)
        
        # Filter by threshold
        filtered_chunks = [
            chunk for chunk in reranked 
            if chunk["relevance_score"] >= self.retrieval_config["rerank_threshold"]
        ]
        
        # Ensure we have at least some results
        if not filtered_chunks and reranked:
            filtered_chunks = reranked[:self.retrieval_config["top_k"]]
        
        return filtered_chunks[:self.retrieval_config["top_k"]]
    
    def _optimize_context(self, chunks: List[Dict], max_tokens: int) -> List[Dict]:
        """
        Optimize context window by selecting and ordering chunks.
        
        Args:
            chunks: List of ranked chunks
            max_tokens: Maximum tokens for context
            
        Returns:
            List[Dict]: Optimized context chunks
        """
        # Sort by relevance score
        sorted_chunks = sorted(chunks, key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Estimate token count (rough approximation)
        def estimate_tokens(text):
            # Rough estimate: 4 chars per token
            return len(text) // 4
        
        # Select chunks up to max_tokens
        selected_chunks = []
        current_tokens = 0
        
        for chunk in sorted_chunks:
            chunk_tokens = estimate_tokens(chunk["content"])
            
            if current_tokens + chunk_tokens <= max_tokens:
                selected_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                # If we can't fit the whole chunk, consider truncating
                if not selected_chunks:  # If no chunks selected yet, take a truncated version
                    truncated_length = max(1, int(max_tokens * 4))  # Convert tokens to chars
                    truncated_content = chunk["content"][:truncated_length] + "..."
                    truncated_chunk = chunk.copy()
                    truncated_chunk["content"] = truncated_content
                    truncated_chunk["truncated"] = True
                    selected_chunks.append(truncated_chunk)
                break
        
        return selected_chunks
    
    def build_prompt(
        self, 
        query: str, 
        context_chunks: List[Dict], 
        template: str = "default",
        chat_history: List[Dict] = []
    ) -> Dict:
        """
        Construct a prompt for the LLM using templates and context.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            template: Template name to use
            chat_history: Previous conversation history
            
        Returns:
            Dict: Formatted prompt for LLM
        """
        # Get the template
        prompt_template = self.templates.get(template, self.templates["default"])
        
        # Format context
        formatted_context = ""
        for i, chunk in enumerate(context_chunks):
            source_info = f"[Source {i+1}"
            if "doc_id" in chunk.get("metadata", {}):
                source_info += f": {chunk['metadata']['doc_id']}"
            if "source" in chunk.get("metadata", {}):
                source_info += f" from {chunk['metadata']['source']}"
            source_info += "]"
            
            formatted_context += f"{source_info}\n{chunk['content']}\n\n"
        
        # Format chat history if available
        chat_context = ""
        if chat_history:
            chat_context = "Previous conversation:\n"
            for exchange in chat_history[-3:]:  # Include last 3 exchanges
                if "query" in exchange and "response" in exchange:
                    chat_context += f"User: {exchange['query']}\nAssistant: {exchange['response']}\n\n"
        
        # Format the prompt
        formatted_prompt = prompt_template.format(
            context=formatted_context,
            query=query,
            chat_history=chat_context
        )
        
        # Create the full prompt with system message
        prompt = {
            "system": self.generation_config["system_prompt"],
            "user": formatted_prompt
        }
        
        return prompt
    
    async def _generate_response(self, prompt: Dict) -> Dict:
        """
        Generate a response from the LLM service.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Dict: LLM response
        """
        try:
            # Format for Ollama API
            ollama_payload = {
                "model": self.generation_config["model"],
                "messages": [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ],
                "options": {
                    "temperature": self.generation_config["temperature"],
                    "top_p": self.generation_config["top_p"],
                    "num_predict": self.generation_config["max_tokens"]
                },
                "stream": False  # Disable streaming to get a single JSON response
            }
            
            # Call Ollama API
            response = await self.http_client.post(
                f"{self.llm_service}/api/chat",
                json=ollama_payload
            )
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    return {
                        "response": result["message"]["content"],
                        "model": self.generation_config["model"],
                        "finish_reason": result.get("done", True),
                        "raw_response": result
                    }
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {str(e)}")
                    logger.debug(f"Raw response: {response.text[:500]}...")  # Log first 500 chars
                    return {
                        "response": "I'm sorry, I encountered an error parsing the response.",
                        "error": f"JSON parsing error: {str(e)}"
                    }
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return {
                    "response": "I'm sorry, I encountered an error generating a response.",
                    "error": f"API error: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "I'm sorry, I encountered an error generating a response.",
                "error": str(e)
            }
    
    def validate_response(self, response: str, context: List[Dict], query: str) -> Dict:
        """
        Validate the response for quality and factual accuracy.
        
        Args:
            response: Generated response
            context: Context chunks used for generation
            query: Original user query
            
        Returns:
            Dict: Validation results
        """
        # Initialize validation results
        validation = {
            "is_valid": True,
            "confidence_score": 0.8,  # Default confidence
            "hallucination_score": 0.0,
            "source_support": 1.0,
            "issues": []
        }
        
        # 1. Check response length
        if len(response) < 10:
            validation["is_valid"] = False
            validation["issues"].append("Response too short")
            validation["confidence_score"] *= 0.5
        
        # 2. Check for source support
        context_text = " ".join([chunk["content"] for chunk in context])
        
        # Simple hallucination detection - check if response contains facts not in context
        # This is a simplified approach - in production, use more sophisticated methods
        response_sentences = [s.strip() for s in response.split(".") if s.strip()]
        supported_sentences = 0
        
        for sentence in response_sentences:
            # Skip very short sentences
            if len(sentence) < 15:
                continue
                
            # Check if key parts of the sentence appear in context
            words = sentence.lower().split()
            key_words = [w for w in words if len(w) > 4 and w not in ["about", "would", "should", "could"]]
            
            if key_words:
                matches = sum(1 for word in key_words if word in context_text.lower())
                support_score = matches / len(key_words)
                
                if support_score >= 0.5:
                    supported_sentences += 1
                elif support_score < 0.3:
                    # Potential hallucination
                    validation["hallucination_score"] += (1 - support_score) / len(response_sentences)
        
        # Calculate source support score
        if response_sentences:
            validation["source_support"] = supported_sentences / len(response_sentences)
        
        # 3. Check for query relevance
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        query_overlap = len(query_terms.intersection(response_terms)) / len(query_terms)
        
        if query_overlap < 0.3:
            validation["issues"].append("Low query relevance")
            validation["confidence_score"] *= 0.8
        
        # 4. Calculate overall confidence score
        validation["confidence_score"] = (
            validation["confidence_score"] * 
            (0.7 + 0.3 * validation["source_support"]) * 
            (1.0 - validation["hallucination_score"])
        )
        
        # 5. Final validation decision
        if (validation["hallucination_score"] > self.quality_config["max_hallucination_score"] or
            validation["confidence_score"] < self.quality_config["min_confidence"]):
            validation["is_valid"] = False
        
        return validation
    
    def extract_citations(self, response: str, chunks: List[Dict]) -> List[Dict]:
        """
        Extract citations linking response claims to source chunks.
        
        Args:
            response: Generated response
            chunks: Context chunks used for generation
            
        Returns:
            List[Dict]: Extracted citations
        """
        citations = []
        
        # Simple citation extraction - look for content from chunks in the response
        for i, chunk in enumerate(chunks):
            chunk_content = chunk["content"]
            
            # Look for significant phrases from the chunk in the response
            phrases = self._extract_significant_phrases(chunk_content)
            
            for phrase in phrases:
                if len(phrase) > 10 and phrase in response:
                    citation = {
                        "text": phrase,
                        "source_chunk_id": i,
                        "source_metadata": chunk.get("metadata", {}),
                        "char_start": response.find(phrase),
                        "char_end": response.find(phrase) + len(phrase)
                    }
                    citations.append(citation)
        
        return citations
    
    def _extract_significant_phrases(self, text: str, min_length: int = 5, max_phrases: int = 3) -> List[str]:
        """
        Extract significant phrases from text.
        
        Args:
            text: Source text
            min_length: Minimum phrase length in words
            max_phrases: Maximum number of phrases to extract
            
        Returns:
            List[str]: Extracted phrases
        """
        # Split into sentences
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        
        phrases = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) >= min_length:
                # For simplicity, use the whole sentence if it's not too long
                if len(words) <= 20:
                    phrases.append(sentence)
                else:
                    # Otherwise, take a substring
                    mid_point = len(words) // 2
                    phrase = " ".join(words[mid_point-min_length:mid_point+min_length])
                    phrases.append(phrase)
        
        # Return up to max_phrases
        return phrases[:max_phrases]
    
    def _log_rag_analytics(self, query: str, collection: str, result: Dict) -> None:
        """
        Log RAG pipeline analytics.
        
        Args:
            query: User query
            collection: Collection name
            result: Processing result
        """
        try:
            # Create analytics entry
            analytics = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "collection": collection,
                "response_length": len(result["response"]),
                "context_chunks": len(result["context"]),
                "confidence_score": result["confidence_score"],
                "processing_time": result["processing_time"]
            }
            
            # In a real implementation, you might store this in a database
            # For now, we'll just log it
            logger.info(f"RAG analytics: {json.dumps(analytics)}")
        except Exception as e:
            logger.error(f"Failed to log RAG analytics: {str(e)}")

# Example usage
async def example_usage():
    """Example usage of the RAG pipeline."""
    # Initialize vector store
    from .vector_store import VectorStore
    vector_store = VectorStore(host="localhost", port=8000)
    
    # Initialize RAG pipeline
    rag = RAGPipeline(
        vector_store=vector_store,
        llm_service="http://localhost:11434",
        config={
            "model": "llama2",
            "temperature": 0.7
        }
    )
    
    # Process a query
    collection_name = "huggingface_collection"
    query = "What are the key provisions of the Americans with Disabilities Act?"
    
    result = await rag.process_query(
        query=query,
        collection=collection_name,
        domain="legal"
    )
    
    print(f"Query: {query}")
    print(f"Response: {result['response']}")
    print(f"Confidence: {result['confidence_score']:.2f}")
    print(f"Processing time: {result['processing_time']:.2f} seconds")
    
    # Print citations
    if result["citations"]:
        print("\nCitations:")
        for citation in result["citations"]:
            print(f"- {citation['text']} (Source: {citation['source_metadata'].get('source', 'Unknown')})")

if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())
