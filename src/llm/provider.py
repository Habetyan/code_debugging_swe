"""
LLM Provider Module

Handles communication with LLM APIs through OpenRouter.
Includes response caching and rate limit handling.
"""

import os
import time
import hashlib
import json
from pathlib import Path
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
import diskcache

load_dotenv()


class LLMProvider:
    """
    LLM provider that communicates with various models through OpenRouter API.
    Uses OpenAI SDK compatibility for seamless integration.
    """
    
    DEFAULT_MODEL = "deepseek/deepseek-chat"
    FALLBACK_MODELS = [
        "meta-llama/llama-3-70b-instruct",
        "mistralai/mistral-7b-instruct",
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_dir: str = "cache/llm",
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY in .env file."
            )
        
        self.model = model or self.DEFAULT_MODEL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.max_retries = max_retries
        
        # Setup disk cache
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.cache = diskcache.Cache(str(cache_path))
    
    def _get_cache_key(self, prompt: str, model: str, temperature: float) -> str:
        """Generate a unique cache key for the request."""
        content = f"{model}:{temperature}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        use_cache: bool = True,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt/query
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            use_cache: Whether to use disk cache
            
        Returns:
            Generated text response
        """
        cache_key = self._get_cache_key(
            f"{system_prompt or ''}|{prompt}",
            self.model,
            temperature
        )
        
        # Check cache
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Retry with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                result = response.choices[0].message.content
                
                # Cache the result
                if use_cache:
                    self.cache[cache_key] = result
                
                return result
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"API error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"LLM API failed after {self.max_retries} attempts: {e}")
    
    def clear_cache(self):
        """Clear the response cache."""
        self.cache.clear()
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "volume_bytes": self.cache.volume(),
        }
