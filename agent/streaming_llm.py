"""Async streaming LLM wrapper for DeepSeek API."""

import os
from typing import AsyncGenerator, Optional
from openai import AsyncOpenAI
from setting import deepseek_api_key


async def call_llm_stream(
    system_prompt: str,
    user_prompt: str,
    api_key: str = deepseek_api_key,
    model: str = "deepseek-chat",
    temperature: float = 1.0,
) -> AsyncGenerator[str, None]:
    """
    Call DeepSeek API with streaming enabled.
    Yields text chunks as they arrive.
    """
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            stream=True,
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        yield f"\n\n[Error: {str(e)}]"


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    api_key: str = deepseek_api_key,
    model: str = "deepseek-chat",
    temperature: float = 1.0,
) -> str:
    """
    Non-streaming call — collects full response.
    """
    chunks = []
    async for chunk in call_llm_stream(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        api_key=api_key,
        model=model,
        temperature=temperature,
    ):
        chunks.append(chunk)
    return "".join(chunks)
