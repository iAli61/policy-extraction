# filepath: /home/azureuser/policy-extraction/lightRag/flamingo.py
import os
import asyncio
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
    if not api_key:
        api_key = os.getenv("FLAMINGO_API_KEY")

    client = AsyncFlamingoLLMClient(
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
        base_url=base_url or "https://api.flamingo.ai/v1",
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        subscription_key=os.getenv("SUBSCRIPTION_KEY"),
        tenant=os.getenv("TENANT"),
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )

    if not response or not response.choices:
        raise Exception("Invalid response from Flamingo API")

    return response.choices[0].message.content

async def test_flamingo_funcs():
    result = await flamingo_complete_if_cache("flamingo-model", "How are you?")
    print("flamingo_complete_if_cache: ", result)

async def main():
    try:
        await test_flamingo_funcs()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())