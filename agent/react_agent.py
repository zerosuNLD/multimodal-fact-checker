"""ReAct agent loop cho fact-checker.

Vòng lặp:
  1. Gọi LLM với system prompt + conversation history
  2. Parse ✿FUNCTION✿ / ✿ARGS✿ → thực thi tool → thêm ✿RESULT✿
  3. Lặp lại cho đến khi LLM xuất ra ✿RETURN✿
  4. Trả về nội dung ✿RETURN✿ là báo cáo cuối

MAX_ITERATIONS: giới hạn số lần gọi tool để tránh vòng lặp vô tận.
"""

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from agent.tools import execute_tool
from agent.streaming_llm import call_llm_stream
from agent.events import emit_event
from llm.prompt.react_agent_prompt import (
    LANGUAGE_NAMES,
    build_agent_prompt,
    build_initial_input,
)

MAX_ITERATIONS = 8  # tối đa 8 lần gọi tool


# ── Helpers ───────────────────────────────────────────────────────────
def _parse_function_call(text: str) -> tuple[str | None, dict | None]:
    """Trích xuất (tool_name, args_dict) từ LLM output.
    Tìm ✿FUNCTION✿ và ✿ARGS✿ trong văn bản.
    """
    fn_match   = re.search(r"✿FUNCTION✿\s*:\s*(\w+)", text)
    args_match = re.search(r"✿ARGS✿\s*:\s*(\{.*?\})", text, re.DOTALL)

    if not fn_match:
        return None, None

    tool_name = fn_match.group(1).strip()
    args = {}
    if args_match:
        try:
            args = json.loads(args_match.group(1))
        except json.JSONDecodeError:
            # Thử làm sạch chuỗi JSON
            raw = args_match.group(1).replace("'", '"')
            try:
                args = json.loads(raw)
            except Exception:
                args = {}

    return tool_name, args


def _extract_return(text: str) -> str | None:
    """Trích xuất nội dung sau ✿RETURN✿."""
    match = re.search(r"✿RETURN✿\s*:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _has_return(text: str) -> bool:
    return bool(re.search(r"✿RETURN✿", text))


# ── Main agent loop ────────────────────────────────────────────────────
async def run_react_agent(
    claim: str,
    image_paths: list[str],
    language: str = "en",
) -> AsyncGenerator[str, None]:
    """Chạy ReAct agent và stream output ra console.

    Yields từng dòng text: bao gồm cả THOUGHT, FUNCTION, RESULT, RETURN.
    Trả về chuỗi rỗng khi kết thúc.
    """
    language_name = LANGUAGE_NAMES.get(language, "English")
    prompt_tmpl   = build_agent_prompt(language_name)
    system_msg    = prompt_tmpl.format_messages(input="")[0].content  # lấy system

    user_input    = build_initial_input(claim, image_paths)

    # Lịch sử hội thoại — list[dict] theo OpenAI messages format
    messages: list[dict] = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_input},
    ]

    loop          = asyncio.get_running_loop()
    final_report  = ""

    for iteration in range(MAX_ITERATIONS + 1):
        # ── Gọi LLM, stream output ──────────────────────────────
        collected = []
        async for chunk in _call_llm_messages_stream(messages):
            print(chunk, end="", flush=True)
            await emit_event("token", {"node": "react_agent", "text": chunk})
            collected.append(chunk)
        llm_response = "".join(collected)
        print()  # newline sau khi stream xong

        # Thêm response của assistant vào lịch sử
        messages.append({"role": "assistant", "content": llm_response})

        # ── Kiểm tra ✿RETURN✿ ──────────────────────────────────
        if _has_return(llm_response):
            final_report = _extract_return(llm_response) or llm_response
            break

        # ── Parse tool call ─────────────────────────────────────
        tool_name, args = _parse_function_call(llm_response)
        if not tool_name:
            # LLM không gọi tool cũng không có RETURN → thêm hint
            hint = "\n✿RESULT✿: Không có tool nào được gọi. Hãy gọi một tool hoặc đưa ra ✿RETURN✿."
            messages.append({"role": "user", "content": hint})
            continue

        if iteration >= MAX_ITERATIONS:
            messages.append({
                "role": "user",
                "content": "\n✿RESULT✿: Đã đạt giới hạn số lần gọi tool. Hãy đưa ra ✿RETURN✿ ngay bây giờ.",
            })
            continue

        # ── Thực thi tool trong thread pool (blocking I/O) ──────
        print(f"\n[Tool: {tool_name}] Args: {args}")
        await emit_event("step", {"node": tool_name.upper(), "status": "running", "output": f"Đang gọi công cụ {tool_name}..."})
        with ThreadPoolExecutor(max_workers=1) as pool:
            tool_result = await loop.run_in_executor(
                pool, lambda: execute_tool(tool_name, args)
            )

        # Rút gọn result nếu quá dài để không làm ngập context
        result_text = _truncate_result(tool_result, max_chars=4000)
        print(f"[Result preview]: {result_text[:200]}...\n")
        await emit_event("step", {"node": tool_name.upper(), "status": "completed", "output": f"Công cụ {tool_name} đã trả về kết quả."})

        # Thêm RESULT vào lịch sử như message của user (ReAct pattern)
        messages.append({
            "role": "user",
            "content": f"✿RESULT✿:\n{result_text}",
        })

    return final_report


def _truncate_result(text: str, max_chars: int = 4000) -> str:
    """Cắt kết quả tool nếu quá dài."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


# ── Streaming LLM với messages list ──────────────────────────────────
async def _call_llm_messages_stream(
    messages: list[dict],
    model: str = "deepseek-chat",
    temperature: float = 1.0,
) -> AsyncGenerator[str, None]:
    """Gọi LLM với full messages list (multi-turn), stream response."""
    from openai import AsyncOpenAI
    from setting import deepseek_api_key

    client = AsyncOpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com",
    )
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"\n[LLM Error: {e}]"
