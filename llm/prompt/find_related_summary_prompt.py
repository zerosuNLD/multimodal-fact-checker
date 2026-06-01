"""Prompt lọc các summary liên quan đến ảnh."""

import json
from llm.prompt._template import ChatPromptTemplate

prompt_raw = {
    "system": """Bạn là một hệ thống phân tích dữ liệu tự động. Nhiệm vụ của bạn là đánh giá một Summary tham chiếu với danh sách các Summary khác được đánh số.

## Quy tắc đầu ra BẮT BUỘC

1. Chỉ chọn các Index của những Summary có thông tin **trực tiếp liên quan** đến Summary tham chiếu (cùng chủ đề, hỗ trợ, mở rộng hoặc chia sẻ ngữ cảnh với nó).
2. Đầu ra DUY NHẤT là một mảng số nguyên chứa các Index đã chọn.
3. Các số phải khớp chính xác với Index trong phần `<danh_sach_summary>`. Không bịa Index không tồn tại.
4. Nếu không có Summary nào liên quan, trả về mảng rỗng: `[]`
5. **KHÔNG** giải thích, **KHÔNG** dùng markdown.

Ví dụ đầu ra hợp lệ: `[0, 2, 5]` hoặc `[]` hoặc `[1]`""",

    "human": """<summary_tham_chieu>
{reference}
</summary_tham_chieu>

<danh_sach_summary>
{summaries}
</danh_sach_summary>

Đầu ra:""",
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_raw["system"]),
        ("human",  prompt_raw["human"]),
    ]
)


def generate_find_related_summary_prompt(highest_summary: dict, summaries: list[dict]) -> list:
    """Trả về [SYSTEM_PROMPT, USER_PROMPT] tương thích với call_llm hiện tại."""
    reference_text = highest_summary.get("summary", "")
    summaries_text = "\n".join(
        f"Index [{i}]: {item.get('summary', str(item))}"
        for i, item in enumerate(summaries)
    )
    messages = prompt.format_messages(reference=reference_text, summaries=summaries_text)
    return [messages[0].content, messages[1].content]