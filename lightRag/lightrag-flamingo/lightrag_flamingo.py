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
model_name="llama3"
from dotenv import load_dotenv
load_dotenv()

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def flamingo_llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await flamingo_complete_if_cache(
        model=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
        base_url=os.getenv("BASE_URL"),
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        subscription_key=os.getenv("SUBSCRIPTION_KEY"),
        tenant_id=os.getenv("TENANT_ID"),
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-mpnet-base-v2")
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


async def initialize_flamingo_rag(working_dir=WORKING_DIR):
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=flamingo_llm_model_func,
        llm_model_name=model_name,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def run_lightrag(
        working_dir: str = WORKING_DIR,
        file_path:str = "./markdown_files/20241119Placing Slip.md",
        query:str = "What are the top themes in this story?",
        mode:str = "naive",
        top_k:int = 10,
        response_type:str = "Single Paragraph",
        ):
    # Setup logging
    logger.setLevel("DEBUG")
    set_verbose_debug(True)
    await test_funcs()
    try:
        # Initialize RAG instance
        rag = await initialize_flamingo_rag(working_dir)

        with open(file_path, "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        answer = await rag.aquery(query, param=QueryParam(
                    mode=mode,
                    response_type=response_type,
                    top_k=top_k
                    )
            )
        print(answer)

        context = await rag.aquery(query, param=QueryParam(
                    mode=mode,
                    only_need_context=True,
                    top_k=top_k
                    )
            )

        return answer, context
        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(run_lightrag())

