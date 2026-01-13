"""Rate Limiter with Exponential Backoff for API calls.

This module provides a rate limiter to handle 429 errors gracefully
by automatically retrying with exponential backoff.
"""

import time
import random
from functools import wraps
from typing import Callable, Any


def with_retry(
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """Decorator that adds retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to prevent thundering herd
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    
                    # Check if it's a rate limit error (429)
                    is_rate_limit = (
                        "429" in error_str or 
                        "too many requests" in error_str or
                        "rate limit" in error_str or
                        "quota" in error_str or
                        "resource_exhausted" in error_str
                    )
                    
                    if not is_rate_limit or attempt == max_retries:
                        raise e
                    
                    last_exception = e
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    print(f"⏳ Rate limit hit. Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
            
            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


async def with_retry_async(
    func: Callable,
    *args,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    **kwargs
) -> Any:
    """Async version of retry logic with exponential backoff.
    
    Args:
        func: Async function to call
        *args: Arguments to pass to the function
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        Result of the function call
    """
    import asyncio
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            
            is_rate_limit = (
                "429" in error_str or 
                "too many requests" in error_str or
                "rate limit" in error_str or
                "quota" in error_str or
                "resource_exhausted" in error_str
            )
            
            if not is_rate_limit or attempt == max_retries:
                raise e
            
            last_exception = e
            
            delay = min(
                base_delay * (2.0 ** attempt),
                max_delay
            )
            delay = delay * (0.5 + random.random())
            
            print(f"⏳ Rate limit hit. Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
            await asyncio.sleep(delay)
    
    if last_exception:
        raise last_exception
