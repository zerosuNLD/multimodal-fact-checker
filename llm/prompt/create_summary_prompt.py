"""Prompt tổng hợp summary từ chunks bài báo."""

from llm.prompt._template import ChatPromptTemplate

# ── Prompt dùng khi có cả caption ảnh + chunks ──────────────────────
_summary_with_caption_raw = {
    "system": """Bạn là một nhà phân tích tin tức chuyên nghiệp. Nhiệm vụ của bạn là tổng hợp caption ảnh và các đoạn văn bản trích xuất thành một bản tóm tắt tin tức mạch lạc.

## Nguyên tắc

1. **Ưu tiên nguồn:** Các đoạn trích (chunks) là nguồn thông tin chính. Caption ảnh chỉ là ngữ cảnh hỗ trợ.
2. **Tường thuật liền mạch:** Kết hợp caption và chunks thành một câu chuyện logic, liên kết.
3. **Lọc thông tin:** Chỉ giữ lại chi tiết trực tiếp liên quan đến sự kiện chính. Bỏ thông tin lặp lại hoặc không liên quan.
4. **Xử lý mâu thuẫn:** Nếu có mâu thuẫn giữa caption và chunks, ưu tiên chunks và đề cập tự nhiên trong bài.
5. **Không bịa đặt:** Chỉ dùng thông tin có trong caption hoặc chunks. Không suy diễn hay thêm chi tiết.
6. **Định dạng:** Viết 1–2 đoạn văn ngắn gọn (khoảng 120–180 từ). Không dùng tiêu đề, gạch đầu dòng hay phần riêng biệt.""",

    "human": """<caption_anh>
{caption}
</caption_anh>

<doan_trich>
{chunks}
</doan_trich>

Tổng hợp thành bản tin ngắn gọn:""",
}

prompt_with_caption = ChatPromptTemplate.from_messages(
    [
        ("system", _summary_with_caption_raw["system"]),
        ("human",  _summary_with_caption_raw["human"]),
    ]
)


# ── Prompt dùng khi chỉ có chunks (không có caption) ────────────────
_summary_text_only_raw = {
    "system": _summary_with_caption_raw["system"],   # cùng system prompt

    "human": """<doan_trich>
{chunks}
</doan_trich>

Tổng hợp thành bản tin ngắn gọn:""",
}

prompt_text_only = ChatPromptTemplate.from_messages(
    [
        ("system", _summary_text_only_raw["system"]),
        ("human",  _summary_text_only_raw["human"]),
    ]
)


# ── Adapter functions (giữ interface cũ cho call_llm) ───────────────
def generate_summary_prompt(caption_image: str, related_chunks: list[str]) -> list:
    """Trả về [SYSTEM_PROMPT, USER_PROMPT] với caption + chunks."""
    chunks_text = "\n".join(f"- Đoạn {i+1}: {c}" for i, c in enumerate(related_chunks))
    messages = prompt_with_caption.format_messages(caption=caption_image, chunks=chunks_text)
    return [messages[0].content, messages[1].content]


def generate_summary_prompt_text_only(related_chunks: list[str]) -> list:
    """Trả về [SYSTEM_PROMPT, USER_PROMPT] chỉ với chunks."""
    chunks_text = "\n".join(f"- Đoạn {i+1}: {c}" for i, c in enumerate(related_chunks))
    messages = prompt_text_only.format_messages(chunks=chunks_text)
    return [messages[0].content, messages[1].content]