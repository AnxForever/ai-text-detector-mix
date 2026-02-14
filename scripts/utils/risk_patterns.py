"""Pattern definitions for risk auditing and dataset triage."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, Pattern


@dataclass(frozen=True)
class RiskPatternSet:
    """Holds compiled hard/soft patterns used by risk scripts."""

    hard_remove: Dict[str, Pattern[str]]
    soft_flag: Dict[str, Pattern[str]]


def _compile_many(items: Iterable[tuple[str, str]]) -> Dict[str, Pattern[str]]:
    """Compile named regex items."""
    return {name: re.compile(pattern, re.IGNORECASE) for name, pattern in items}


PATTERNS = RiskPatternSet(
    hard_remove=_compile_many(
        [
            (
                "instruction_leak_cn",
                (
                    r"(?:分析请求|逐句分析|改进思路|用户让我润色|直接输出润色后的文本|"
                    r"目标\s*[:：]\s*润色)"
                ),
            ),
            (
                "assistant_meta_cn",
                r"(?:好的，用户|用户希望|我需要先|我将会|接下来我会)",
            ),
            (
                "assistant_meta_en",
                r"(?:as an ai language model|i need to analyze|let me break this down)",
            ),
            (
                "roleplay_prompt",
                (
                    r"(?m)^\s*(?:system|assistant|user)\s*:"
                    r"|(?:角色设定|请你扮演|你现在扮演|作为(?:一个)?AI|"
                    r"你是(?:一名|一个)?(?:AI|助手|语言模型))"
                ),
            ),
            (
                "template_header",
                r"(?m)^\s*(?:题目|要求|提示|output)\s*[:：]",
            ),
        ]
    ),
    soft_flag=_compile_many(
        [
            ("markdown_table", r"\|.{2,}\|"),
            ("markdown_list", r"(?:^|\n)\s*(?:[-\*+]|\d+\.)\s+"),
            ("markdown_code", r"```[\s\S]*?```|`[^`\n]{1,64}`"),
            ("heading_style", r"(?:^|\n)\s*#{1,6}\s+"),
            ("excessive_punctuation", r"[!！?？]{3,}"),
            ("overlong_text", r"^.{7000,}$"),
        ]
    ),
)

