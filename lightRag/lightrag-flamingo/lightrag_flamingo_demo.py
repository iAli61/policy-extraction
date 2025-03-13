import os
import asyncio
from flamingo import flamingo_complete_if_cache # Importing the flamingo_complete_if_cache function
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status
from sentence_transformers import SentenceTransformer
from lightrag.utils import logger, set_verbose_debug

WORKING_DIR = "./test"
from dotenv import load_dotenv
load_dotenv()

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def flamingo_llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await flamingo_complete_if_cache(
        model="llama3",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        api_key=os.getenv("API_KEY"),
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
        base_url=os.getenv("BASE_URL"),
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        subscription_key=os.getenv("SUBSCRIPTION_KEY"),
        tenant=os.getenv("TENANT_ID"),
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await flamingo_llm_model_func("How are you?")
    print("flamingo_llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
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
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    # Setup logging
    logger.setLevel("DEBUG")
    set_verbose_debug(True)
    await test_funcs()
    try:
        # Initialize RAG instance
        rag = await initialize_flamingo_rag()

        with open("./markdown_files/20241119Placing Slip.md", "r", encoding="utf-8") as f:
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