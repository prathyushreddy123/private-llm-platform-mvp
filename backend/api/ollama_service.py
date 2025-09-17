"""
Ollama Service for LLM Integration.

This module provides a service for interacting with Ollama LLM models,
including model management, inference, caching, and error handling.
"""

import os
import json
import time
import logging
import asyncio
import httpx
from typing import Dict, List, Optional, Union, AsyncGenerator, Tuple, Any
import tiktoken
import redis
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ollama_service")

class OllamaService:
    """Service for interacting with Ollama LLM models."""
    
    def __init__(
        self,
        base_url: str = os.environ.get("OLLAMA_API_URL", "http://localhost:11434"),
        cache_ttl: int = int(os.environ.get("OLLAMA_CACHE_TTL", 3600)),
        default_model: str = os.environ.get("OLLAMA_DEFAULT_MODEL", "mistral:7b"),
        fallback_model: str = os.environ.get("OLLAMA_FALLBACK_MODEL", "mistral:7b-instruct"),
        request_timeout: int = int(os.environ.get("OLLAMA_REQUEST_TIMEOUT", 120)),
        max_retries: int = int(os.environ.get("OLLAMA_MAX_RETRIES", 3)),
    ):
        """
        Initialize the Ollama service.
        
        Args:
            base_url: Base URL for the Ollama API
            cache_ttl: Time-to-live for cached responses in seconds
            default_model: Default model to use for inference
            fallback_model: Fallback model to use if default fails
            request_timeout: Timeout for API requests in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.base_url = base_url
        self.cache_ttl = cache_ttl
        self.default_model = default_model
        self.fallback_model = fallback_model
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        
        # Initialize Redis client for caching if REDIS_URL is set
        redis_url = os.environ.get("REDIS_URL")
        self.redis_client = redis.from_url(redis_url) if redis_url else None
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(timeout=request_timeout)
        
        # Initialize tokenizer for token counting
        # Default to cl100k_base which works for many models
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"Initialized OllamaService with base_url={base_url}, default_model={default_model}")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def health_check(self) -> bool:
        """
        Check if Ollama API is healthy and responsive.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    async def list_models(self) -> List[Dict]:
        """
        List all available models in Ollama.
        
        Returns:
            List of model information dictionaries
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return response.json().get("models", [])
            else:
                logger.error(f"Failed to list models: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    async def ensure_model_available(self, model_name: str) -> bool:
        """
        Ensure that a model is available, downloading it if necessary.
        
        Args:
            model_name: Name of the model to check/download
            
        Returns:
            bool: True if model is available, False otherwise
        """
        models = await self.list_models()
        model_names = [model.get("name") for model in models]
        
        if model_name in model_names:
            logger.info(f"Model {model_name} is already available")
            return True
        
        logger.info(f"Model {model_name} not found, attempting to download")
        try:
            # Pull the model
            response = await self.client.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name}
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully downloaded model {model_name}")
                return True
            else:
                logger.error(f"Failed to download model {model_name}: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {str(e)}")
            return False
    
    def _get_cache_key(self, prompt: str, model: str, params: Dict) -> str:
        """
        Generate a cache key for a request.
        
        Args:
            prompt: The prompt text
            model: The model name
            params: Additional parameters
            
        Returns:
            str: Cache key
        """
        # Create a deterministic representation of the request
        cache_data = {
            "prompt": prompt,
            "model": model,
            "params": params
        }
        return f"ollama:response:{hash(json.dumps(cache_data, sort_keys=True))}"
    
    async def _get_cached_response(self, prompt: str, model: str, params: Dict) -> Optional[Dict]:
        """
        Get a cached response if available.
        
        Args:
            prompt: The prompt text
            model: The model name
            params: Additional parameters
            
        Returns:
            Optional[Dict]: Cached response or None
        """
        if not self.redis_client:
            return None
        
        cache_key = self._get_cache_key(prompt, model, params)
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            try:
                return json.loads(cached_data)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode cached response for key {cache_key}")
        
        return None
    
    async def _cache_response(self, prompt: str, model: str, params: Dict, response: Dict) -> None:
        """
        Cache a response.
        
        Args:
            prompt: The prompt text
            model: The model name
            params: Additional parameters
            response: The response to cache
        """
        if not self.redis_client:
            return
        
        cache_key = self._get_cache_key(prompt, model, params)
        try:
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(response)
            )
            logger.debug(f"Cached response with key {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache response: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def manage_context_window(self, messages: List[Dict], max_tokens: int = 4096) -> List[Dict]:
        """
        Truncate or compress context to fit within token window.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            List[Dict]: Truncated message list
        """
        # Count tokens in all messages
        token_count = 0
        messages_with_tokens = []
        
        for msg in messages:
            content = msg.get("content", "")
            tokens = self.count_tokens(content)
            messages_with_tokens.append({
                "message": msg,
                "tokens": tokens
            })
            token_count += tokens
        
        # If we're within the limit, return the original messages
        if token_count <= max_tokens:
            return messages
        
        # We need to truncate
        logger.info(f"Context exceeds {max_tokens} tokens ({token_count}). Truncating...")
        
        # Always keep the system message if present
        result = []
        remaining_tokens = max_tokens
        
        # First, add any system messages
        for item in messages_with_tokens:
            if item["message"].get("role") == "system":
                result.append(item["message"])
                remaining_tokens -= item["tokens"]
        
        # Then add the most recent messages that fit
        for item in reversed(messages_with_tokens):
            if item["message"].get("role") != "system":  # Skip system messages already added
                if item["tokens"] <= remaining_tokens:
                    result.insert(0, item["message"])  # Insert at beginning to maintain order
                    remaining_tokens -= item["tokens"]
                else:
                    # This message is too large, we could truncate it but for simplicity we'll skip
                    continue
        
        logger.info(f"Truncated context to {max_tokens - remaining_tokens} tokens")
        return result
    
    @retry(
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying request after error (attempt {retry_state.attempt_number})"
        )
    )
    async def generate(
        self,
        prompt: str,
        model: str = None,
        stream: bool = False,
        use_cache: bool = True,
        **kwargs
    ) -> Union[Dict, AsyncGenerator[Dict, None]]:
        """
        Generate text using the Ollama API.
        
        Args:
            prompt: The prompt text
            model: The model to use (defaults to self.default_model)
            stream: Whether to stream the response
            use_cache: Whether to use caching
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Union[Dict, AsyncGenerator[Dict, None]]: Response data
        """
        start_time = time.time()
        model = model or self.default_model
        
        # Check if model is available
        model_available = await self.ensure_model_available(model)
        if not model_available:
            if model != self.fallback_model:
                logger.warning(f"Model {model} not available, falling back to {self.fallback_model}")
                return await self.generate(prompt, self.fallback_model, stream, use_cache, **kwargs)
            else:
                raise ValueError(f"Neither primary model nor fallback model {self.fallback_model} are available")
        
        # Count input tokens
        input_tokens = self.count_tokens(prompt)
        logger.info(f"Request: model={model}, input_tokens={input_tokens}, stream={stream}")
        
        # Check cache for non-streaming requests
        if use_cache and not stream:
            cached_response = await self._get_cached_response(prompt, model, kwargs)
            if cached_response:
                logger.info(f"Cache hit for prompt with model {model}")
                return cached_response
        
        # Prepare request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        
        try:
            if stream:
                return self._stream_response(prompt, model, payload)
            else:
                return await self._generate_complete_response(prompt, model, payload, use_cache)
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            if model != self.fallback_model:
                logger.warning(f"Falling back to {self.fallback_model} after error")
                return await self.generate(prompt, self.fallback_model, stream, use_cache, **kwargs)
            raise
        finally:
            duration = time.time() - start_time
            logger.info(f"Generation completed in {duration:.2f}s")
    
    async def _generate_complete_response(
        self,
        prompt: str,
        model: str,
        payload: Dict,
        use_cache: bool
    ) -> Dict:
        """
        Generate a complete (non-streaming) response.
        
        Args:
            prompt: The prompt text
            model: The model name
            payload: Request payload
            use_cache: Whether to use caching
            
        Returns:
            Dict: Response data
        """
        response = await self.client.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Count output tokens
        output_tokens = self.count_tokens(result.get("response", ""))
        result["metrics"] = {
            "input_tokens": self.count_tokens(prompt),
            "output_tokens": output_tokens,
            "total_tokens": self.count_tokens(prompt) + output_tokens
        }
        
        # Cache the response if enabled
        if use_cache:
            await self._cache_response(prompt, model, payload, result)
        
        return result
    
    async def _stream_response(
        self,
        prompt: str,
        model: str,
        payload: Dict
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream the response from Ollama.
        
        Args:
            prompt: The prompt text
            model: The model name
            payload: Request payload
            
        Yields:
            Dict: Response chunks
        """
        async with self.client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
            response.raise_for_status()
            
            full_response = ""
            async for chunk in response.aiter_lines():
                if not chunk:
                    continue
                    
                try:
                    data = json.loads(chunk)
                    full_response += data.get("response", "")
                    yield data
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode chunk: {chunk}")
            
            # Yield final metrics
            output_tokens = self.count_tokens(full_response)
            yield {
                "done": True,
                "metrics": {
                    "input_tokens": self.count_tokens(prompt),
                    "output_tokens": output_tokens,
                    "total_tokens": self.count_tokens(prompt) + output_tokens
                }
            }
    
    async def embed_text(
        self,
        text: str,
        model: str = "nomic-embed-text",
        **kwargs
    ) -> Dict:
        """
        Generate embeddings for text using Ollama.
        
        Args:
            text: The text to embed
            model: The embedding model to use
            **kwargs: Additional parameters
            
        Returns:
            Dict: Embedding response
        """
        # Check if model is available
        model_available = await self.ensure_model_available(model)
        if not model_available:
            raise ValueError(f"Embedding model {model} is not available")
        
        payload = {
            "model": model,
            "prompt": text,
            **kwargs
        }
        
        try:
            response = await self.client.post(f"{self.base_url}/api/embeddings", json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Add token metrics
            result["metrics"] = {
                "input_tokens": self.count_tokens(text)
            }
            
            return result
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

# Example usage
async def example_usage():
    async with OllamaService() as service:
        # Check health
        healthy = await service.health_check()
        print(f"Ollama service healthy: {healthy}")
        
        # List models
        models = await service.list_models()
        print(f"Available models: {models}")

        # Warmup the model with a simple request
        print("Warming up the model (this may take a minute)...")
        try:
            await service.generate("Hello", max_tokens=10)
            print("Model warmed up successfully!")
        except Exception as e:
            print(f"Warmup failed: {str(e)}")
        
        # Generate text
        response = await service.generate(
            "Explain the concept of retrieval-augmented generation in simple terms.",
            temperature=0.7,
            max_tokens=500
        )
        print(f"Generated response: {response['response']}")
        print(f"Metrics: {response['metrics']}")
        
        # Stream response
        print("\nStreaming response:")
        stream_generator = await service.generate(
            "Write a short poem about artificial intelligence.",
            stream=True
        )
        async for chunk in stream_generator:
            if "response" in chunk:
                print(chunk["response"], end="", flush=True)
        print("\n")

if __name__ == "__main__":
    asyncio.run(example_usage())
