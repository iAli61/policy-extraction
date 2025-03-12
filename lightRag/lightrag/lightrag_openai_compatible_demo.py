import os
import asyncio
from flamingo_client import AsyncFlamingoLLMClient
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status

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
    return await client.chat.completions.create(
        model="flamingo-model",
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )


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


async def initialize_flamingo_rag():
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=flamingo_llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=flamingo_embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    try:
        # Initialize RAG instance
        rag = await initialize_flamingo_rag()

        with open("./book.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # Perform naive search
        print(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="naive")
            )
        )

        # Perform local search
        print(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="local")
            )
        )

        # Perform global search
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="global"),
            )
        )

        # Perform hybrid search
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="hybrid"),
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())