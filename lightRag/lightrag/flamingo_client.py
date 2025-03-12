import os
import asyncio
from flamingo import FlamingoLLMClient, AsyncFlamingoLLMClient
import numpy as np

async def flamingo_complete_if_cache(
    client: AsyncFlamingoLLMClient,
    prompt: str,
    system_prompt: str = None,
    history_messages: list[dict] = None,
    **kwargs
) -> str:
    if history_messages is None:
        history_messages = []
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model="flamingo-model",  # Replace with the actual model name
        messages=messages,
        **kwargs
    )

    if not response or not response.choices:
        raise Exception("Invalid response from Flamingo API")

    return response.choices[0].message.content

async def flamingo_embed(
    client: AsyncFlamingoLLMClient,
    texts: list[str],
    model: str = "flamingo-embedding-model"  # Replace with the actual embedding model name
) -> np.ndarray:
    response = await client.embeddings.create(
        model=model,
        input=texts
    )
    return np.array([dp.embedding for dp in response.data])

# Example usage
async def main():
    client = AsyncFlamingoLLMClient(
        subscription_id="your_subscription_id",
        base_url="https://api.flamingo.ai",
        client_id="your_client_id",
        client_secret="your_client_secret",
        subscription_key="your_subscription_key",
        tenant="your_tenant"
    )

    prompt = "What are the main themes in the story?"
    result = await flamingo_complete_if_cache(client, prompt)
    print("Flamingo Completion Result:", result)

    texts = ["This is a test sentence."]
    embeddings = await flamingo_embed(client, texts)
    print("Flamingo Embeddings Result:", embeddings)

if __name__ == "__main__":
    asyncio.run(main())