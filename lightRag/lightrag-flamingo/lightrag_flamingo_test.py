# filepath: /home/azureuser/policy-extraction/lightRag/lightrag/lightrag_flamingo_demo.py
import os
import asyncio
from flamingo_client import FlamingoLLMClient, AsyncFlamingoLLMClient
from dotenv import load_dotenv
load_dotenv()

WORKING_DIR = "./test"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    client = AsyncFlamingoLLMClient(
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
        base_url=os.getenv("BASE_URL"),
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        subscription_key=os.getenv("SUBSCRIPTION_KEY"),
        tenant=os.getenv("TENANT_ID"),
    )
    return await client.chat.completions.create(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )

async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

async def main():
    try:
        await test_funcs()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())