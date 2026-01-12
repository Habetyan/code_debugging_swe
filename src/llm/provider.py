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
    Supports multiple API keys for fallback when one runs out of credits.
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
        # Load all API keys from environment
        self.api_keys = self._load_api_keys(api_key)
        if not self.api_keys:
            raise ValueError(
                "No OpenRouter API keys found. "
                "Set OPENROUTER_API_KEY (and optionally OPENROUTER_API_KEY1, KEY2, etc.) in .env file."
            )
        
        self.current_key_index = 0
        self.model = model or self.DEFAULT_MODEL
        self.max_retries = max_retries
        
        # Initialize client with first key
        self.client = self._create_client(self.api_keys[0])
        
        # Setup disk cache
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.cache = diskcache.Cache(str(cache_path))
    
    def _load_api_keys(self, provided_key: Optional[str]) -> list[str]:
        """Load all available API keys from environment."""
        keys = []
        
        # Add provided key first
        if provided_key:
            keys.append(provided_key)
        
        # Load from environment variables
        base_key = os.getenv("OPENROUTER_API_KEY")
        if base_key and base_key not in keys:
            keys.append(base_key)
        
        # Load numbered keys (KEY1, KEY2, KEY3, etc.)
        for i in range(1, 10):
            key = os.getenv(f"OPENROUTER_API_KEY{i}")
            if key and key not in keys:
                keys.append(key)
        
        return keys
    
    def _create_client(self, api_key: str) -> OpenAI:
        """Create OpenAI client with given API key."""
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    
    def _switch_to_next_key(self) -> bool:
        """Switch to next API key. Returns False if no more keys available."""
        self.current_key_index += 1
        if self.current_key_index < len(self.api_keys):
            self.client = self._create_client(self.api_keys[self.current_key_index])
            print(f"[Fallback] Switched to API key #{self.current_key_index + 1}")
            return True
        return False
    
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
        
        # Try with all API keys first (for the primary model), then fallback models
        last_error = None
        
        # First, try all API keys with the primary model
        while True:
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
                    last_error = e
                    error_str = str(e).lower()
                    
                    # If it's a credit/payment error, try next API key
                    if '402' in str(e) or 'credit' in error_str or 'payment' in error_str:
                        print(f"[API Key #{self.current_key_index + 1}] Credit limit reached...")
                        if self._switch_to_next_key():
                            break  # Retry with new key
                        else:
                            # No more keys, try fallback models
                            break
                    
                    # For other errors, retry with backoff
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"API error: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        # Try next API key
                        if self._switch_to_next_key():
                            break
                        else:
                            break
            else:
                # Inner loop completed without break (success or exhausted retries)
                continue
            
            # Check if we've exhausted all keys
            if self.current_key_index >= len(self.api_keys):
                break
            continue
        
        # All API keys exhausted, try fallback models
        for fallback_model in self.FALLBACK_MODELS:
            print(f"[Fallback] Trying model: {fallback_model}")
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=fallback_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    result = response.choices[0].message.content
                    
                    if use_cache:
                        self.cache[cache_key] = result
                    
                    print(f"[Fallback] Success with {fallback_model}")
                    return result
                    
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                    continue
        
        # All options exhausted
        raise RuntimeError(f"All API keys and fallback models failed. Last error: {last_error}")
    
    def clear_cache(self):
        """Clear the response cache."""
        self.cache.clear()
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "volume_bytes": self.cache.volume(),
        }
