# filepath: /home/azureuser/policy-extraction/lightRag/flamingo.py
import os
import asyncio
from flamingo_client import FlamingoLLMClient, AsyncFlamingoLLMClient
import numpy as np

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def flamingo_llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    client = AsyncFlamingoLLMClient(
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
        base_url=os.getenv("FLAMINGO_BASE_URL"),
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        subscription_key=os.getenv("SUBSCRIPTION_KEY"),
        tenant=os.getenv("TENANT"),
    )
    
    response = await client.chat.completions.create(
        model="flamingo-model",
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    
    return response.choices[0].message.content


async def flamingo_embedding_func(texts: list[str]) -> np.ndarray:
    client = AsyncFlamingoLLMClient(
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
        base_url=os.getenv("FLAMINGO_BASE_URL"),
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        subscription_key=os.getenv("SUBSCRIPTION_KEY"),
        tenant=os.getenv("TENANT"),
    )
    
    response = await client.embeddings.create(
        model="flamingo-embedding-model",
        input=texts,
    )
    
    return np.array([dp.embedding for dp in response.data])


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await flamingo_embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await flamingo_llm_model_func("How are you?")
    print("flamingo_llm_model_func: ", result)

    result = await flamingo_embedding_func(["How are you?"])
    print("flamingo_embedding_func: ", result)


async def initialize_flamingo():
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    # Additional initialization logic can be added here

    return embedding_dimension


async def main():
    try:
        # Initialize Flamingo instance
        await initialize_flamingo()

        # Perform tests
        await test_funcs()
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())