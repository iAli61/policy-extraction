import os
import asyncio
import numpy as np
from flamingo_client import FlamingoLLMClient, AsyncFlamingoLLMClient

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def flamingo_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str = None,
    history_messages: list[dict] = None,
    api_key: str = None,
    base_url: str = None,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    
    # Create client
    client = AsyncFlamingoLLMClient(
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
        base_url=base_url or "https://api.flamingo.ai/v1",
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        subscription_key=os.getenv("SUBSCRIPTION_KEY"),
        tenant=os.getenv("TENANT"),
    )

    # Construct messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Make API call
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )

    if not response or not response.choices:
        raise Exception("Invalid response from Flamingo API")

    return response.choices[0].message.content

async def flamingo_embed(
    texts: list[str],
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    **kwargs
) -> np.ndarray:
    """
    Get embeddings for the provided texts using the Flamingo API.
    
    Note: If Flamingo doesn't support embeddings directly, you might need
    to use a different service or approach for embeddings.
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