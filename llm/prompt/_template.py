"""Minimal ChatPromptTemplate replacement — không cần langchain.

Dùng nội bộ trong dự án để giữ cấu trúc prompt_raw / prompt
mà không phụ thuộc thêm package.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class _FormattedMessage:
    """Message object sau khi format — có cả dict-style và attribute-style access."""
    role: str
    content: str

    def __getitem__(self, key: str):
        return getattr(self, key)


@dataclass
class _Message:
    role: Literal["system", "human", "ai"]
    template: str

    def format(self, **kwargs) -> _FormattedMessage:
        """Replace {key} placeholders một cách thủ công — tránh lỗi KeyError
        khi template có các ký tự { } không phải placeholder (vd: JSON example)."""
        content = self.template
        for key, value in kwargs.items():
            content = content.replace(f"{{{key}}}", str(value))
        return _FormattedMessage(role=self.role, content=content)


class ChatPromptTemplate:
    """Minimal drop-in cho langchain.prompts.ChatPromptTemplate."""

    def __init__(self, messages: list[_Message]):
        self._messages = messages

    @classmethod
    def from_messages(cls, raw: list[tuple[str, str]]) -> "ChatPromptTemplate":
        """raw: list of (role, template_string)."""
        return cls([_Message(role=r, template=t) for r, t in raw])

    def format_messages(self, **kwargs) -> list[dict]:
        return [m.format(**kwargs) for m in self._messages]
