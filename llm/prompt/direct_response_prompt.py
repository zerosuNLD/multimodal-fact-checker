"""Prompt phản hồi trực tiếp cho agent fact-check.

Sử dụng định dạng ✿THOUGHT✿ / ✿FUNCTION✿ / ✿ARGS✿ / ✿RESULT✿ / ✿RETURN✿
để agent hiển thị quá trình suy luận rõ ràng.
"""

import os
from llm.prompt._template import ChatPromptTemplate

LANGUAGE_NAMES = {
    "en": "English",
    "vi": "tiếng Việt",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}

# ── Shared system prompt template ────────────────────────────────────
_SYSTEM_TEMPLATE = """\
Bạn là một chuyên gia kiểm chứng thông tin. Nhiệm vụ của bạn là đọc các bằng chứng được cung cấp và viết một báo cáo tóm gọn, dễ hiểu.

## Định dạng suy luận

Sử dụng định dạng sau để thể hiện quá trình phân tích:

```
⯸THOUGHT⯸: [Nhận xét bước đầu — bằng chứng nào có giá trị nhất?]

⯸THOUGHT⯸: [Tiếp tục phân tích cho đến khi đủ thông tin...]

⯸RETURN⯸: [Báo cáo cuối cùng]
```

## Yêu cầu đầu ra (⯸RETURN⯸)

Viết một hoặc hai đoạn văn tự nhiên tóm tắt những gì bằng chứng cho thấy và kết luận cuối cùng của bạn. Cuối cùng liệt kê các nguồn đã dùng:

Nguồn tham khảo:
[1] <url>
[2] <url>
...

## Ngườn ngữ

QUAN TRỌNG: Toàn bộ phần ⯸RETURN⯸ phải viết bằng {language_name}.

## Nguyên tắc

- Viết tự nhiên như một nhà báo, KHÔNG dùng tiêu đề hay mục phân chia.
- Mỗi nhận định thực tế nên có trích dẫn nội tuyến [X] tham chiếu đến nguồn.
- KHÔNG bọa đặt thông tin không có trong bằng chứng.
- Nếu bằng chứng yếu hoặc mâu thuẫn, nêu rõ sự không chắc chắn.\
"""

# ── Scenario-specific return structure instructions ──────────────────
_RETURN_INSTRUCTIONS = {
    "image_only": """

Viết 1–2 đoạn văn tự nhiên mô tả những gì bằng chứng hình ảnh cho thấy, kèm kết luận của bạn. Cuối cùng liệt kê nguồn:

Nguồn tham khảo:
[1] <url>
[2] <url>""",

    "text_only": """

Viết 1–2 đoạn văn tự nhiên tóm tắt bằng chứng và đưa ra kết luận (REAL / FAKE / MISLEADING / UNVERIFIED). Cuối cùng liệt kê nguồn:

Nguồn tham khảo:
[1] <url>
[2] <url>""",

    "both": """

Viết 1–2 đoạn văn tự nhiên tóm tắt bằng chứng (cả ảnh lẫn văn bản) và đưa ra kết luận (REAL / FAKE / MISLEADING / UNVERIFIED). Cuối cùng liệt kê nguồn:

Nguồn tham khảo:
[1] <url>
[2] <url>""",
}

# ── ChatPromptTemplate definitions ───────────────────────────────────
def _make_prompt(scenario: str) -> ChatPromptTemplate:
    # Ghép scenario_instructions vào cuối system template
    # language_name vẫn là placeholder {language_name}, được fill qua format_messages()
    scenario_block = _RETURN_INSTRUCTIONS[scenario]
    # Thay thế thủ công để tránh dùng str.format() trên toàn bộ template
    system = _SYSTEM_TEMPLATE.replace("{scenario_instructions}", scenario_block)
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{user_content}"),
        ]
    )

_prompts = {
    "image_only": _make_prompt("image_only"),
    "text_only":  _make_prompt("text_only"),
    "both":       _make_prompt("both"),
}


# ── Evidence formatters ───────────────────────────────────────────────
def _format_links(links_list: list) -> str:
    return "\n".join(f"[{i}]: {link}" for i, link in enumerate(links_list, start=1))


def _format_image_evidence(filtered_results, links_list: list) -> str:
    texts = []
    items = filtered_results.items() if isinstance(filtered_results, dict) else filtered_results
    for item_group in items:
        if not (isinstance(item_group, (list, tuple)) and len(item_group) >= 2):
            continue
        image_path, summaries = item_group[0], item_group[1]
        image_name = os.path.basename(image_path)
        for item in summaries:
            url     = item.get("url", "")
            title   = item.get("title", "")
            snippet = item.get("snippet", "")
            summary = item.get("summary", "")
            if not summary or not url:
                continue
            try:
                idx = links_list.index(url) + 1
                texts.append(
                    f"--- Nguồn [{idx}] ---\n"
                    f"Ảnh: {image_name}\n"
                    f"Tiêu đề: {title}\n"
                    f"URL: {url}\n"
                    f"Tóm tắt nhanh: {snippet}\n"
                    f"Nội dung: {summary}"
                )
            except ValueError:
                continue
    return "\n\n".join(texts) if texts else "[KHÔNG CÓ BẰNG CHỨNG HÌNH ẢNH]"


def _format_text_evidence(claim_results, links_list: list) -> str:
    url_to_data: dict = {}
    if claim_results and isinstance(claim_results, dict):
        for query, results in claim_results.items():
            if not results:
                continue
            for item in results:
                url     = item.get("url", "")
                title   = item.get("title", "")
                snippet = item.get("snippet", "")
                summary = item.get("summary", "")
                if url and summary:
                    url_to_data[url] = {
                        "title": title,
                        "snippet": snippet,
                        "summary": summary,
                    }

    if not url_to_data:
        return "[KHÔNG CÓ BẰNG CHỨNG VĂN BẢN]"

    parts = []
    for i, link in enumerate(links_list, start=1):
        if link in url_to_data:
            d = url_to_data[link]
            parts.append(
                f"[Nguồn {i}]\n"
                f"Tiêu đề: {d['title']}\n"
                f"URL: {link}\n"
                f"Tóm tắt nhanh: {d['snippet']}\n"
                f"Nội dung: {d['summary']}\n"
            )
    return "\n".join(parts) if parts else "[KHÔNG CÓ BẰNG CHỨNG VĂN BẢN]"


# ── Public API ────────────────────────────────────────────────────────
def generate_direct_response_prompt(
    claim: str,
    image_results: dict,
    claim_results: dict,
    links_list: list,
    language: str,
    scenario: str = "both",
) -> list:
    """Trả về [SYSTEM_PROMPT, USER_PROMPT] theo scenario.

    Dùng ChatPromptTemplate với định dạng ✿THOUGHT✿/✿RETURN✿.
    """
    language_name = LANGUAGE_NAMES.get(language, "English")
    prompt_tmpl   = _prompts.get(scenario, _prompts["both"])
    links_text    = _format_links(links_list)

    # Build user content block
    if scenario == "image_only":
        user_content = (
            f"<Nguon_Danh_So>\n{links_text}\n</Nguon_Danh_So>\n\n"
            f"<Bang_Chung_Hinh_Anh>\n{_format_image_evidence(image_results, links_list)}\n</Bang_Chung_Hinh_Anh>\n\n"
            "Hãy phân tích hình ảnh được tải lên:"
        )
    elif scenario == "text_only":
        user_content = (
            f"<Claim>\n{claim or '[KHÔNG CÓ CLAIM]'}\n</Claim>\n\n"
            f"<Nguon_Danh_So>\n{links_text}\n</Nguon_Danh_So>\n\n"
            f"<Bang_Chung_Van_Ban>\n{_format_text_evidence(claim_results, links_list)}\n</Bang_Chung_Van_Ban>\n\n"
            "Hãy tạo phản hồi kiểm chứng thông tin:"
        )
    else:  # both
        user_content = (
            f"<Claim>\n{claim or '[KHÔNG CÓ CLAIM]'}\n</Claim>\n\n"
            f"<Nguon_Danh_So>\n{links_text}\n</Nguon_Danh_So>\n\n"
            f"<Bang_Chung_Hinh_Anh>\n{_format_image_evidence(image_results, links_list)}\n</Bang_Chung_Hinh_Anh>\n\n"
            f"<Bang_Chung_Van_Ban>\n{_format_text_evidence(claim_results, links_list)}\n</Bang_Chung_Van_Ban>\n\n"
            "Hãy tạo phản hồi kiểm chứng thông tin:"
        )

    messages = prompt_tmpl.format_messages(
        language_name=language_name,
        user_content=user_content,
    )

    system = messages[0].content
    human  = messages[1].content
    return [system, human]
