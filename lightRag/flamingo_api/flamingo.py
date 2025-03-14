import os
import asyncio
import numpy as np
import logging
import sys

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)

from flamingo_client import AsyncFlamingoLLMClient, AuthException
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
) 
from lightrag.utils import (
    verbose_debug,
    VERBOSE_DEBUG,
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
    logger,
)
from lightrag.api import __api_version__
from typing import Any, Union

class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InvalidResponseError, AuthException)
    ),
)
async def flamingo_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    base_url: str | None = None,
    subscription_id: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    subscription_key: str | None = None,
    tenant_id: str | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    
    # Set default headers similar to OpenAI client
    default_headers = {
        "User-Agent": f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) LightRAG/{__api_version__}",
        "Content-Type": "application/json",
    }
    
    # Set Flamingo logger level to INFO when VERBOSE_DEBUG is off
    if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
        logging.getLogger("flamingo_client").setLevel(logging.INFO)
    
    # Create async client with proper authentication - similar to OpenAI pattern
    try:
        flamingo_async_client = AsyncFlamingoLLMClient(
            subscription_id=subscription_id or os.getenv("SUBSCRIPTION_ID", ""),
            base_url=base_url or os.getenv("BASE_URL", ""),
            client_id=client_id or os.getenv("CLIENT_ID", ""),
            client_secret=client_secret or os.getenv("CLIENT_SECRET", ""),
            subscription_key=subscription_key or os.getenv("SUBSCRIPTION_KEY", ""),
            tenant=tenant_id or os.getenv("TENANT_ID", ""),
        )
    except Exception as e:
        logger.error(f"Failed to initialize Flamingo client: {e}")
        raise APIConnectionError(f"Failed to initialize Flamingo client: {e}")

    # Handle parameters not used by Flamingo API
    kwargs.pop("hashing_kv", None)
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    
    # Construct messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Logging similar to OpenAI client
    logger.debug("===== Sending Query to Flamingo LLM =====")
    logger.debug(f"Model: {model}   Base URL: {base_url}")
    logger.debug(f"Additional kwargs: {kwargs}")
    verbose_debug(f"Query: {prompt}")
    verbose_debug(f"System prompt: {system_prompt}")

    # Make API call with proper error handling
    try:
        # Handle response_format parameter similar to OpenAI
        if "response_format" in kwargs:
            # Flamingo implements the response_format parameter directly in chat.completions.create
            response = await flamingo_async_client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
        else:
            response = await flamingo_async_client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
    except AuthException as e:
        logger.error(f"Flamingo Authentication Error: {e}")
        raise APIConnectionError(f"Authentication error: {e}")
    except Exception as e:
        error_str = str(e)
        if "connection" in error_str.lower():
            logger.error(f"Flamingo API Connection Error: {e}")
            raise APIConnectionError(str(e))
        elif "rate" in error_str.lower() or "limit" in error_str.lower():
            logger.error(f"Flamingo API Rate Limit Error: {e}")
            raise RateLimitError(str(e))
        elif "timeout" in error_str.lower():
            logger.error(f"Flamingo API Timeout Error: {e}")
            raise APITimeoutError(str(e))
        else:
            logger.error(
                f"Flamingo API Call Failed,\nModel: {model},\nParams: {kwargs}, Got: {e}"
            )
            raise

    # Handle streaming responses
    if hasattr(response, "__aiter__"):
        async def inner():
            try:
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content is None:
                        continue
                    if r"\u" in content:
                        content = safe_unicode_decode(content.encode("utf-8"))
                    yield content
            except Exception as e:
                logger.error(f"Error in stream response: {str(e)}")
                raise
        return inner()
    else:
        # Handle regular responses
        if (
            not response
            or not response.choices
            or len(response.choices) == 0
            or not hasattr(response.choices[0], "message")
            or not hasattr(response.choices[0].message, "content")
        ):
            logger.error("Invalid response from Flamingo API")
            raise InvalidResponseError("Invalid response from Flamingo API")

        content = response.choices[0].message.content

        if not content or content.strip() == "":
            logger.error("Received empty content from Flamingo API")
            raise InvalidResponseError("Received empty content from Flamingo API")

        # Handle unicode escapes
        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))
            
        # Handle keyword extraction if enabled
        # if keyword_extraction:
        #     content = locate_json_string_body_from_string(content)
            
        return content

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def flamingo_embed(
    texts: list[str],
    model: str = "flamingo-embedding-model",  # Update with actual model name
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    """
    Get embeddings for the provided texts using the Flamingo API.
    
    Note: Implementation depends on whether Flamingo supports embeddings directly.
    If not, this function needs to be modified or removed.
    """
    # Since Flamingo client doesn't support embeddings as shown in the code,
    # this is a placeholder. You might need to implement this differently
    # or use a different service for embeddings.
    
    # For example, you might use:
    # - A different API client for embeddings
    # - A local embedding model
    # - Another service's embedding API
    
    raise NotImplementedError("Flamingo API does not support embeddings directly. Please use a different embedding service.")

async def test_flamingo_funcs():
    result = await flamingo_complete_if_cache("flamingo-model", "How are you?")
    print("flamingo_complete_if_cache: ", result)
    
    # Uncomment if embeddings are supported
    # embed_result = await flamingo_embed(["How are you?"])
    # print("flamingo_embed: ", embed_result)

async def main():
    try:
        await test_flamingo_funcs()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())