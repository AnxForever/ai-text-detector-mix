"""Lightweight project-knowledge retrieval for thesis defense Q&A.

This module builds a small in-process retrieval index from the current
repository so the API can answer questions about the project using local
documents and selected code files.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import ZipFile

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency in lean envs
    PdfReader = None

from scripts.utils.paths import PATHS, PROJECT_ROOT

PROJECT_QA_MIN_RELEVANCE_SCORE = float(os.getenv("DC_PROJECT_QA_MIN_RELEVANCE_SCORE", "0.12"))

DEFAULT_SOURCE_PATTERNS = (
    "docs/project/DEFENSE_KB_v2.md",
    "docs/project/DEFENSE_DEMO_SCRIPT.md",
    "docs/project/DEFENSE_KB_CURATED.md",
    "docs/project/DEFENSE_CURRENT_STATUS.md",
    "docs/project/ADVISOR_ACADEMIC_QA.md",
    "docs/project/RISK_IMPLEMENTATION_*.md",
    "docs/thesis/project_technical_deep_dive.md",
    "docs/thesis/theoretical_foundations.md",
    "docs/thesis/thesis_data_reference.md",
    "docs/thesis/chapter5_experiments_filled.md",
    "models/bert_v11c_boundary_fix/README.md",
    "models/bert_v11c_boundary_fix/eval_comparison.json",
    "models/bert_v11c_boundary_fix/eval_perclass.json",
    "models/bert_v11c_boundary_fix/training_log.json",
    "evaluation_results/*baseline_results.json",
    "evaluation_results/benchmark_inference_results.json",
    "datasets/eval/splits/v1/README.md",
    "README.md",
    "QUICKSTART.md",
    "CLAUDE.md",
    "AGENTS.md",
    "*.docx",
    "docs/*.md",
    "docs/project/*.md",
    "docs/plans/*.md",
    "docs/thesis/*.md",
    "docs/thesis/*.docx",
    "docs/thesis/build/*.md",
    "api/README.md",
    "api/CLAUDE.md",
    "api/api.py",
    "frontend/README.md",
    "frontend/CLAUDE.md",
    "models/CLAUDE.md",
    "evaluation_results/*.md",
    "evaluation_results/*.txt",
    "evaluation_results/*.json",
    "models/*/eval_comparison.json",
    "models/*/eval_perclass.json",
    "models/*/training_log.json",
    "models/*/README.md",
    "datasets/project_qa_uploads/**/*",
)

CODE_SYMBOL_SOURCE_PATTERNS = (
    "api/**/*.py",
    "scripts/**/*.py",
    "frontend/app/**/*.ts",
    "frontend/app/**/*.tsx",
    "frontend/components/**/*.ts",
    "frontend/components/**/*.tsx",
    "frontend/lib/**/*.ts",
)

SUPPORTED_SOURCE_SUFFIXES = {".md", ".txt", ".json", ".py", ".docx", ".pdf", ".pptx"}

_QUESTION_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?；;])\s+|\n+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_STOPWORDS = {
    "这个",
    "那个",
    "一个",
    "可以",
    "什么",
    "为什么",
    "怎么",
    "如何",
    "一下",
    "项目",
    "系统",
    "请问",
    "一下子",
    "about",
    "what",
    "why",
    "how",
}
_KNOWN_QUERY_PHRASES = (
    "数据治理",
    "data-centric",
    "数据清洗",
    "训练集",
    "来源覆盖",
    "项目介绍",
    "研究目标",
    "应用价值",
    "工程价值",
    "准确率",
    "召回率",
    "精确率",
    "混淆矩阵",
    "误报",
    "漏报",
    "temperature scaling",
    "ece",
    "边界定位",
    "边界检测",
    "混合文本",
    "数据泄露",
    "过拟合",
    "泛化",
    "分布外",
    "bert",
    "gpt",
    "llama",
    "模型大小",
    "训练数据",
    "训练时长",
    "GPU",
    "显存",
    "基线方法",
    "对比方法",
    "英文文本",
)

_GENERAL_OUT_OF_SCOPE_PATTERNS = (
    "天气",
    "气温",
    "下雨",
    "几点",
    "现在时间",
    "今天几号",
    "股价",
    "汇率",
    "今天新闻",
    "最新新闻",
    "订机票",
    "chatgpt好用",
    "你觉得chatgpt",
    "快排",
    "排序算法",
    "react组件",
    "帮我写一个",
    "帮我写个",
)

_PROJECT_SCOPE_HINTS = (
    "项目",
    "系统",
    "模型",
    "bert",
    "ai文本",
    "ai 文本",
    "检测",
    "训练",
    "数据",
    "论文",
    "答辩",
    "仓库",
    "代码实现",
    "api",
    "fastapi",
    "next.js",
    "前端",
    "后端",
)

_EXCLUDED_SOURCE_PATTERNS = (
    re.compile(r"(^|/)archive(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)docs/thesis/build/", re.IGNORECASE),
    re.compile(r"gpt_context_pack", re.IGNORECASE),
    # Note: 论文章节文件 chapter*_template.md / abstract_template.md
    # 虽然以 _template.md 结尾但实际已填充完整内容（含 Andrew Ng / Data-Centric AI /
    # GPTZero / Confident Learning 等理论锚点），必须纳入 KB。
    # 仅排除明确为空模板占位的文件名（如果未来出现）。
    re.compile(r"evaluation_results/sliding_window_preds_.*\.csv$", re.IGNORECASE),
    re.compile(r"docs/project/final_results\.md$", re.IGNORECASE),
    re.compile(r"docs/project/data_and_models\.md$", re.IGNORECASE),
    re.compile(r"docs/project/presentation\.md$", re.IGNORECASE),
    re.compile(r"docs/project/training_plan\.md$", re.IGNORECASE),
    re.compile(r"docs/plans/defense_qa_preparation\.md$", re.IGNORECASE),
    re.compile(r"docs/experiment_final_results\.md$", re.IGNORECASE),
    re.compile(r"docs/experiment_multi_model_results\.md$", re.IGNORECASE),
    re.compile(r"evaluation_results/comprehensive_evaluation_summary\.md$", re.IGNORECASE),
    re.compile(r"evaluation_results/final_report\.txt$", re.IGNORECASE),
)


@dataclass(slots=True)
class KnowledgeChunk:
    """A single retrievable chunk from the repository."""

    chunk_id: str
    path: str
    title: str
    section: str
    content: str


@dataclass(slots=True)
class KnowledgeHit:
    """A scored retrieval hit."""

    chunk: KnowledgeChunk
    score: float
    excerpt: str


@dataclass(slots=True)
class CodeSymbol:
    """Structured local code symbol for reliable snippet lookup."""

    symbol: str
    path: str
    kind: str
    signature: str | None
    snippet: str


def discover_project_sources(
    root: Path = PROJECT_ROOT,
    patterns: Sequence[str] = DEFAULT_SOURCE_PATTERNS,
) -> list[Path]:
    """Collect the main thesis/project knowledge sources from the repo."""
    seen: set[Path] = set()
    sources: list[Path] = []

    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
                continue
            try:
                normalized_path = path.relative_to(root).as_posix()
            except ValueError:
                normalized_path = path.as_posix()
            if any(pattern.search(normalized_path) for pattern in _EXCLUDED_SOURCE_PATTERNS):
                continue
            if path in seen:
                continue
            seen.add(path)
            sources.append(path)

    for path in list_uploaded_project_sources():
        if path in seen:
            continue
        seen.add(path)
        sources.append(path)

    return sources


def discover_code_symbol_sources(
    root: Path = PROJECT_ROOT,
    patterns: Sequence[str] = CODE_SYMBOL_SOURCE_PATTERNS,
) -> list[Path]:
    """Collect code files that should contribute structured symbol references."""
    seen: set[Path] = set()
    sources: list[Path] = []

    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".py", ".ts", ".tsx"}:
                continue
            try:
                normalized_path = path.relative_to(root).as_posix()
            except ValueError:
                normalized_path = path.as_posix()
            if any(pattern.search(normalized_path) for pattern in _EXCLUDED_SOURCE_PATTERNS):
                continue
            if path in seen:
                continue
            seen.add(path)
            sources.append(path)

    return sources


def _build_code_symbol_snippet(text: str, start_line: int, end_line: int) -> str:
    """Return a compact multi-line snippet around a symbol definition."""
    lines = text.splitlines()
    if not lines:
        return ""

    start_index = max(start_line - 1, 0)
    end_index = min(max(end_line, start_line), len(lines))
    snippet_lines = lines[start_index:end_index]
    snippet = "\n".join(snippet_lines).strip()
    if len(snippet) > 900:
        snippet = snippet[:897].rstrip() + "..."
    return snippet


def extract_python_code_symbols(text: str, path: str) -> list[CodeSymbol]:
    """Extract top-level Python classes/functions using AST."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    symbols: list[CodeSymbol] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        symbol_name = getattr(node, "name", "").strip()
        if not symbol_name:
            continue

        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        signature = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [arg.arg for arg in node.args.args]
            signature = f"{symbol_name}({', '.join(args)})"
        else:
            signature = f"class {symbol_name}"

        end_lineno = getattr(node, "end_lineno", node.lineno)
        snippet = _build_code_symbol_snippet(text, node.lineno, end_lineno)
        if not snippet:
            continue
        symbols.append(
            CodeSymbol(
                symbol=symbol_name,
                path=path,
                kind=kind,
                signature=signature,
                snippet=snippet,
            )
        )
    return symbols


def extract_typescript_code_symbols(text: str, path: str) -> list[CodeSymbol]:
    """Extract exported TS/TSX functions/components with lightweight regex."""
    symbols: list[CodeSymbol] = []
    lines = text.splitlines()
    patterns = [
        (re.compile(r"^export\s+default\s+function\s+(\w+)"), "function"),
        (re.compile(r"^export\s+function\s+(\w+)"), "function"),
        (
            re.compile(
                r"^export\s+const\s+(\w+)\s*=\s*(?:memo\()?(?:function\s+\w+|\([^)]*\)\s*=>)"
            ),
            "function",
        ),
        (re.compile(r"^const\s+(\w+)\s*=\s*memo\(function"), "function"),
        (re.compile(r"^function\s+(\w+)"), "function"),
    ]

    seen: set[str] = set()
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        for pattern, kind in patterns:
            match = pattern.match(stripped)
            if not match:
                continue
            symbol_name = match.group(1)
            if not symbol_name or symbol_name in seen:
                continue
            seen.add(symbol_name)
            snippet = _build_code_symbol_snippet(text, index, min(index + 20, len(lines)))
            symbols.append(
                CodeSymbol(
                    symbol=symbol_name,
                    path=path,
                    kind=kind,
                    signature=symbol_name,
                    snippet=snippet,
                )
            )
            break

    return symbols


def build_code_symbol_index(root: Path = PROJECT_ROOT) -> dict[str, list[CodeSymbol]]:
    """Build a lightweight symbol index for local code references."""
    index: dict[str, list[CodeSymbol]] = {}

    for source_path in discover_code_symbol_sources(root=root):
        try:
            relative_path = source_path.relative_to(root).as_posix()
        except ValueError:
            relative_path = source_path.as_posix()
        text = source_path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue

        if source_path.suffix.lower() == ".py":
            symbols = extract_python_code_symbols(text, relative_path)
        else:
            symbols = extract_typescript_code_symbols(text, relative_path)

        for symbol in symbols:
            key = symbol.symbol.casefold()
            index.setdefault(key, []).append(symbol)

    return index


def read_project_source(path: Path) -> str:
    """Read a UTF-8 text source and normalize blank space lightly."""
    if path.suffix.lower() == ".json":
        return read_project_json_source(path)
    if path.suffix.lower() == ".docx":
        return read_project_docx_source(path)
    if path.suffix.lower() == ".pptx":
        return read_project_pptx_source(path)
    if path.suffix.lower() == ".pdf":
        return read_project_pdf_source(path)

    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\ufeff", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _flatten_json_object(value: Any, prefix: str = "") -> list[str]:
    """Flatten nested JSON into readable key-value text lines."""
    lines: list[str] = []

    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten_json_object(item, next_prefix))
        return lines

    if isinstance(value, list):
        for index, item in enumerate(value):
            next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            lines.extend(_flatten_json_object(item, next_prefix))
        return lines

    normalized_value = value
    if isinstance(value, float):
        normalized_value = round(value, 6)
    lines.append(f"{prefix}: {normalized_value}")
    return lines


def read_project_json_source(path: Path) -> str:
    """Convert important JSON artifacts into retrieval-friendly plain text."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return path.read_text(encoding="utf-8", errors="ignore").strip()

    title = f"file: {path.name}"
    lines = [title]
    lines.extend(_flatten_json_object(payload))
    return "\n".join(lines).strip()


def read_project_docx_source(path: Path) -> str:
    """Extract plain text from a DOCX file without external dependencies."""
    try:
        with ZipFile(path) as archive:
            xml_bytes = archive.read("word/document.xml")
    except Exception:
        return ""

    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        return ""

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
        merged = "".join(texts).strip()
        if merged:
            paragraphs.append(merged)

    header = f"file: {path.name}"
    body = "\n\n".join(paragraphs)
    return f"{header}\n{body}".strip()


def read_project_pptx_source(path: Path) -> str:
    """Extract slide text from a PPTX file without external dependencies."""
    try:
        with ZipFile(path) as archive:
            slide_names = sorted(
                name
                for name in archive.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            )
            if not slide_names:
                return ""

            slides: list[str] = []
            namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
            for slide_name in slide_names:
                root = ElementTree.fromstring(archive.read(slide_name))
                texts = [node.text for node in root.findall(".//a:t", namespace) if node.text]
                merged = " ".join(text.strip() for text in texts if text.strip())
                if merged:
                    slides.append(f"slide {len(slides) + 1}: {merged}")
    except Exception:
        return ""

    header = f"file: {path.name}"
    body = "\n\n".join(slides)
    return f"{header}\n{body}".strip()


def read_project_pdf_source(path: Path) -> str:
    """Extract readable text from a PDF.

    Prefer pypdf when available. Fall back to a lightweight byte-string extractor
    so the agent still gets partial signal in environments where pypdf is absent.
    """
    if PdfReader is not None:
        try:
            reader = PdfReader(str(path))
            pages = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(f"page {page_number}: {text}")
            if pages:
                return f"file: {path.name}\n" + "\n\n".join(pages)
        except Exception:
            pass

    try:
        raw = path.read_bytes()
    except OSError:
        return ""

    matches = re.findall(rb"[\x20-\x7e\xe4-\xe9\xa1-\xbf\x80-\xff]{8,}", raw)
    decoded_parts: list[str] = []
    for match in matches[:400]:
        try:
            text = match.decode("utf-8", errors="ignore").strip()
        except Exception:
            continue
        if text:
            decoded_parts.append(text)

    compact = "\n".join(decoded_parts)
    compact = re.sub(r"\s{2,}", " ", compact).strip()
    if not compact:
        return ""
    return f"file: {path.name}\n{compact}"


def list_uploaded_project_sources() -> list[Path]:
    """List user-uploaded project-qa materials."""
    upload_dir = PATHS.ensure_dir(PATHS.project_qa_uploads_dir)
    return [
        path
        for path in sorted(upload_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SOURCE_SUFFIXES
    ]


def chunk_project_text(text: str, *, chunk_size: int = 900) -> list[str]:
    """Split text into medium-sized chunks for retrieval.

    The chunker favors paragraph boundaries first and only falls back to
    sentence-level splitting when a single paragraph is too large.
    """
    if not text.strip():
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current_parts, current_len
        if current_parts:
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = []
            current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            flush_current()
            sentences = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(paragraph) if s.strip()]
            sentence_bucket: list[str] = []
            sentence_len = 0
            for sentence in sentences or [paragraph]:
                extra_len = len(sentence) + (1 if sentence_bucket else 0)
                if sentence_bucket and sentence_len + extra_len > chunk_size:
                    chunks.append(" ".join(sentence_bucket).strip())
                    sentence_bucket = [sentence]
                    sentence_len = len(sentence)
                else:
                    sentence_bucket.append(sentence)
                    sentence_len += extra_len

            if sentence_bucket:
                chunks.append(" ".join(sentence_bucket).strip())
            continue

        extra_len = len(paragraph) + (2 if current_parts else 0)
        if current_parts and current_len + extra_len > chunk_size:
            flush_current()

        current_parts.append(paragraph)
        current_len += extra_len

    flush_current()
    return chunks


def chunk_markdown_sections(text: str, *, chunk_size: int = 900) -> list[tuple[str, str]]:
    """Split markdown into heading-aware chunks.

    Returns `(section_title, chunk_text)` pairs so retrieval can keep section
    context attached to the chunk, instead of treating all paragraphs equally.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    lines = cleaned.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = "Document"
    current_lines: list[str] = []

    def flush_section() -> None:
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_title, current_lines.copy()))
        current_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()
        heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading_match:
            flush_section()
            current_title = heading_match.group(2).strip()
            continue
        current_lines.append(line)

    flush_section()

    if not sections:
        return [("Document", chunk) for chunk in chunk_project_text(cleaned, chunk_size=chunk_size)]

    chunk_pairs: list[tuple[str, str]] = []
    for section_title, section_lines in sections:
        section_body = "\n".join(section_lines).strip()
        for chunk in chunk_project_text(section_body, chunk_size=chunk_size):
            chunk_pairs.append((section_title, chunk))
    return chunk_pairs


def build_project_chunks(
    root: Path = PROJECT_ROOT,
    patterns: Sequence[str] = DEFAULT_SOURCE_PATTERNS,
) -> list[KnowledgeChunk]:
    """Build retrievable chunks from the repository's main project sources."""
    chunks: list[KnowledgeChunk] = []

    for source_path in discover_project_sources(root=root, patterns=patterns):
        try:
            relative_path = source_path.relative_to(root).as_posix()
        except ValueError:
            relative_path = f"project_qa_uploads/{source_path.name}"
        source_text = read_project_source(source_path)
        if source_path.suffix.lower() == ".md":
            raw_chunks = chunk_markdown_sections(source_text)
        else:
            raw_chunks = [(source_path.stem, chunk) for chunk in chunk_project_text(source_text)]

        for index, (section_title, chunk_text) in enumerate(raw_chunks, start=1):
            contextual_chunk = chunk_text.strip()
            if section_title and section_title.lower() != source_path.stem.lower():
                contextual_chunk = f"section: {section_title}\n{contextual_chunk}"
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{relative_path}#{index}",
                    path=relative_path,
                    title=source_path.stem,
                    section=section_title,
                    content=contextual_chunk,
                )
            )

    return chunks


def _build_excerpt(text: str, *, limit: int = 240) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _build_chunk_search_text(chunk: KnowledgeChunk) -> str:
    """Compose retrieval text with file and section context."""
    parts = [f"title: {chunk.title}"]
    if chunk.section and chunk.section.lower() != chunk.title.lower():
        parts.append(f"section: {chunk.section}")
    parts.append(chunk.content)
    return "\n".join(parts)


def _extract_question_keywords(question: str) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    lowered_question = question.lower()

    def add_keyword(keyword: str) -> None:
        if keyword not in seen:
            keywords.append(keyword)
            seen.add(keyword)

    for phrase in _KNOWN_QUERY_PHRASES:
        if phrase in lowered_question:
            add_keyword(phrase)

    if any(
        phrase in lowered_question
        for phrase in ("介绍一下", "项目介绍", "整体介绍", "概括", "介绍整个项目")
    ):
        for phrase in ("研究目标", "核心方法", "应用价值", "工程价值"):
            add_keyword(phrase)

    for token in _QUESTION_TOKEN_PATTERN.findall(question):
        lowered = token.lower()
        if lowered in _STOPWORDS or lowered in seen:
            continue
        add_keyword(lowered)
    return keywords


def _contains_any(text: str, tokens: Sequence[str]) -> bool:
    """Return whether any token appears in text."""
    return any(token.lower() in text for token in tokens)


def _is_general_out_of_scope_question(question: str) -> bool:
    """Detect questions that should not be answered from the project repository."""
    lowered = question.lower()

    if _contains_any(lowered, _GENERAL_OUT_OF_SCOPE_PATTERNS):
        return True

    code_help = _contains_any(
        lowered, ("帮我看代码", "分析代码", "代码有没有 bug", "写的代码", "debug", "帮我看看这个", "帮我写", "快排", "排序算法")
    )
    if code_help and not _contains_any(lowered, _PROJECT_SCOPE_HINTS):
        return True

    # Detect chat/opinion questions unrelated to the project
    chat_patterns = _contains_any(lowered, ("你觉得chatgpt", "chatgpt好用", "你对gpt的看法", "人工智能会取代", "你觉得 chatgpt", "chatgpt 好用"))
    if chat_patterns and not _contains_any(lowered, _PROJECT_SCOPE_HINTS):
        return True

    return False


def build_project_decline_answer(question: str) -> str | None:
    """Return a direct refusal/unknown answer for questions outside the KB scope."""
    if _is_general_out_of_scope_question(question):
        return (
            "这个问题不属于当前毕业设计项目知识库范围内，我不能从仓库资料里可靠回答。"
            "如果你想练答辩，可以问项目方法、数据集、实验指标、部署实现或局限性。"
        )
    return None


def build_contextual_retrieval_query(
    question: str,
    history: Sequence[dict[str, str]] | None = None,
) -> str:
    """Expand elliptical follow-up questions with recent conversation context."""
    lowered = question.lower()
    additions: list[str] = []

    if history:
        recent_lines: list[str] = []
        for message in list(history)[-4:]:
            content = re.sub(r"\s+", " ", (message.get("content") or "").strip())
            if not content:
                continue
            if len(content) > 120:
                content = content[:117].rstrip() + "..."
            recent_lines.append(content)
        if recent_lines and _contains_any(
            lowered,
            ("它", "这个", "那", "相比", "对比", "别的方法", "别的模型", "哪个更好"),
        ):
            additions.append("最近对话上下文：" + " / ".join(recent_lines))

    if _contains_any(lowered, ("别的方法", "别的模型", "相比", "对比", "差多少")):
        additions.append("意图：基线方法对比 FastText TextCNN DPCNN BERT-BiGRU V11c 召回率 99.28")
    if _contains_any(lowered, ("它", "这个", "那")) and _contains_any(
        lowered, ("比", "方法", "模型")
    ):
        additions.append("意图：本文方法与基线方法对比 BERT-BiGRU FastText TextCNN DPCNN")

    if not additions:
        return question
    return f"{question}\n" + "\n".join(additions)


def _expand_project_query(question: str) -> str:
    """Append controlled synonyms so colloquial questions retrieve the KB, not noisy docs."""
    lowered = question.lower()
    expansions: list[str] = []

    def add(text: str) -> None:
        if text not in expansions:
            expansions.append(text)

    if _contains_any(lowered, ("训练数据", "训练集", "多少条", "哪些来源", "来源覆盖", "数据哪来", "数据来源")):
        add("训练集规模 63,113 来源覆盖 8 大 LLM 家族 46 个具体模型 92 类人类文本")
    if _contains_any(
        lowered, ("不用更新", "更新的预训练", "更强模型", "更大的模型", "为什么不换", "roberta")
    ):
        add("为什么选择 BERT 而不是 RoBERTa GPT LLaMA 更强模型 判别任务 工程成本 可复现 数据治理")
    if _contains_any(lowered, ("模型多大", "模型大小", "显存", "资源占用", "部署效率", "推理速度", "多大")):
        add("模型文件大小 391 MB BERT-base 110M GPU 峰值显存 672 MB 吞吐 127.4 样本/秒 部署效率")
    if _contains_any(lowered, ("gpu", "显卡", "训练环境", "训练时长", "训了多久", "训多久", "训了多长时间")):
        add("实验环境 NVIDIA GeForce RTX 5060 Laptop GPU 8 GB VRAM 最佳模型训练时长 41 分钟")
    if _contains_any(lowered, ("讲讲", "做了啥", "做了什么", "介绍一下", "整个项目", "有啥用", "有什么用", "能干嘛")):
        add("项目介绍 中文 AI 生成文本检测 BERT 微调 分类检测 混合文本 边界定位 工程部署")
    if _contains_any(lowered, ("英文", "英语", "多语")):
        add("英文文本 中文场景 bert-base-chinese 英文需要重新选择预训练模型 数据和评估协议")
    if _contains_any(lowered, ("数据治理", "数据清洗", "清洗", "治理", "帮助")):
        add("Data-Centric AI V10 V11c 数据治理 模板样本 unknown 弱域样本 长文边界 97.69 98.57")
    if _contains_any(lowered, ("死记硬背", "数据泄露", "泄露", "过拟合", "去重")):
        add("数据泄露 无泄露 independent_data 去重 训练未见模型 格式对抗 过拟合")
    if _contains_any(lowered, ("c2", "续写", "一半人写", "一半ai", "人写一段", "混合")):
        add("C2 AI续写 混合文本 [SEP] 边界标记 79.82 93.84 边界定位")
    if _contains_any(
        lowered, ("别的方法", "基线", "对比方法", "fasttext", "textcnn", "bert-bigru")
    ):
        add("基线方法对比 FastText TextCNN DPCNN BERT-BiGRU 本文方法 V11c 召回率 99.28")
    if _contains_any(lowered, ("硬伤", "缺点", "短板", "不足", "局限", "风险", "不靠谱")):
        add("局限性 泛化 风险 过拟合 跨域 边界 外部有效性 不足 改进方向")
    if _contains_any(lowered, ("靠谱", "可信", "可靠", "准不准", "准吗")):
        add("准确率 98.69 校准 ECE 0.0034 可信度 温度校准 无泄露评估集")
    if _contains_any(lowered, ("改进", "怎么改", "下一步", "未来")):
        add("未来工作 改进方向 持续纳入新模型 对抗鲁棒性 可解释性 量化蒸馏 多模态")
    if _contains_any(lowered, ("核心指标", "指标有哪些", "报什么", "报哪些")):
        add("准确率 98.69 三集平均 98.56 独立评估集 98.57 召回率 99.28 ECE 0.0034")
    if _contains_any(lowered, ("2599", "2,599", "主口径", "为什么用这个")):
        add("2599 条无泄露评估集 三个子集 core_v1 independent_data merged_v2 去泄露校验")
    if _contains_any(lowered, ("数据治理", "模型校准", "两者关系")):
        add("Data-Centric AI 数据治理 Temperature Scaling 校准 V10 V11c 模板样本 unknown 弱域")
    if _contains_any(lowered, ("检测能力", "部署效率", "兼顾")):
        add("双层检测架构 分类器 边界检测器 推理速度 127.4 样本/秒 显存 672 MB 部署")

    if not expansions:
        return question
    return f"{question}\n" + "\n".join(expansions)


def _normalize_retrieval_text(text: str) -> str:
    """Normalize text before character n-gram extraction."""
    compact = _WHITESPACE_PATTERN.sub(" ", text).strip().lower()
    return compact


def _char_ngram_counts(text: str, ngram_range: tuple[int, int] = (2, 4)) -> Counter[str]:
    """Build character n-gram counts without external ML dependencies."""
    normalized = _normalize_retrieval_text(text)
    if not normalized:
        return Counter()

    min_n, max_n = ngram_range
    counts: Counter[str] = Counter()
    text_length = len(normalized)
    for n in range(min_n, max_n + 1):
        if text_length < n:
            continue
        for start in range(text_length - n + 1):
            counts[normalized[start : start + n]] += 1
    return counts


def _build_sparse_tfidf_vector(
    counts: Counter[str],
    *,
    doc_freq: dict[str, int],
    doc_count: int,
) -> tuple[dict[str, float], float]:
    """Convert n-gram counts into a sparse TF-IDF vector."""
    if not counts:
        return {}, 0.0

    vector: dict[str, float] = {}
    squared_norm = 0.0
    for token, count in counts.items():
        frequency = doc_freq.get(token)
        if not frequency:
            continue
        tf = 1.0 + math.log(float(count))
        idf = math.log((1.0 + doc_count) / (1.0 + float(frequency))) + 1.0
        weight = tf * idf
        vector[token] = weight
        squared_norm += weight * weight

    return vector, math.sqrt(squared_norm)


def _cosine_similarity_sparse(
    left: dict[str, float],
    right: dict[str, float],
    *,
    left_norm: float,
    right_norm: float,
) -> float:
    """Compute cosine similarity between sparse vectors."""
    if not left or not right or left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0

    if len(left) > len(right):
        left, right = right, left
        left_norm, right_norm = right_norm, left_norm

    dot_product = sum(weight * right.get(token, 0.0) for token, weight in left.items())
    if dot_product <= 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _looks_like_low_value_sentence(sentence: str) -> bool:
    normalized = sentence.strip()
    if len(normalized) < 8:
        return True

    lowered = normalized.lower()
    if any(
        token in lowered
        for token in (
            '"question":',
            '"answer":',
            "def ",
            "class ",
            "import ",
            "return (",
            "if any(",
            "agent_mode ==",
            "lowered for token",
        )
    ):
        return True

    punctuation_count = sum(1 for char in normalized if char in "{}[]()=:_/\\\"'")
    if punctuation_count / max(len(normalized), 1) > 0.12:
        return True

    return False


def _source_path_boost(path: str) -> float:
    lowered = path.lower()
    boost = 0.0

    if "docs/project/defense_kb_v2.md" in lowered:
        boost += 0.30
    if "docs/project/defense_current_status.md" in lowered:
        boost += 0.18
    if "docs/project/defense_kb_curated.md" in lowered:
        boost += 0.26
    if "docs/project/advisor_academic_qa.md" in lowered:
        boost += 0.22
    if lowered.endswith("eval_comparison.json"):
        boost += 0.16
    if lowered.endswith("training_log.json"):
        boost += 0.14
    if "models/claude.md" in lowered:
        boost += 0.12
    if "docs/thesis/project_technical_deep_dive.md" in lowered:
        boost += 0.1
    if "docs/plans/defense_qa_preparation.md" in lowered:
        boost += 0.1
    if "docs/project/final_results.md" in lowered:
        boost += 0.08
    if "evaluation_results/" in lowered:
        boost += 0.06
    if "project_qa_uploads/" in lowered:
        boost += 0.12
    if lowered.endswith("readme.md") or lowered.endswith("claude.md"):
        boost += 0.02

    if "docs/archive/" in lowered or "gpt_context_pack" in lowered or "/pandoc/" in lowered:
        boost -= 0.08

    return boost


def _question_path_boost(question: str, path: str) -> float:
    """Apply question-aware path boosts so defense docs outrank meta docs."""
    lowered_question = question.lower()
    lowered_path = path.lower()
    boost = 0.0
    overview_question = any(
        token in lowered_question
        for token in (
            "30 秒",
            "30秒",
            "介绍整个项目",
            "整体介绍",
            "概括",
            "介绍一下",
            "项目介绍",
        )
    )

    if lowered_path in {
        "api/readme.md",
        "api/claude.md",
        "frontend/claude.md",
        "claude.md",
        "agents.md",
    }:
        boost -= 0.18
    if "model_testing_guide" in lowered_path:
        boost -= 0.2
    if overview_question and "docs/project/defense_demo_script.md" in lowered_path:
        boost -= 0.18
    if overview_question and (
        lowered_path.endswith((".py", ".ts", ".tsx"))
        or lowered_path.startswith(("api/", "scripts/", "frontend/"))
    ):
        boost -= 0.28

    if overview_question:
        if "docs/project/defense_kb_curated.md" in lowered_path:
            boost += 0.22
        if "docs/project/advisor_academic_qa.md" in lowered_path:
            boost += 0.2
        if "docs/project/defense_current_status.md" in lowered_path:
            boost += 0.28
        if lowered_path == "readme.md":
            boost += 0.22
        if "project_technical_deep_dive" in lowered_path:
            boost += 0.14

    if any(
        token in lowered_question
        for token in ("准确率", "指标", "模型", "v11", "v10", "ece", "温度", "多少")
    ):
        if "docs/project/defense_kb_curated.md" in lowered_path:
            boost += 0.3
        if "docs/project/advisor_academic_qa.md" in lowered_path:
            boost += 0.26
        if "docs/project/defense_current_status.md" in lowered_path:
            boost += 0.32
        if lowered_path.endswith("eval_comparison.json"):
            boost += 0.24
        if lowered_path == "frontend/readme.md":
            boost += 0.2
        if lowered_path == "readme.md":
            boost += 0.12

    if any(
        token in lowered_question
        for token in ("训练数据", "训练集规模", "多少条", "来源覆盖", "63,113", "63113")
    ):
        if "docs/project/defense_kb_curated.md" in lowered_path:
            boost += 0.34
        if "docs/project/advisor_academic_qa.md" in lowered_path:
            boost += 0.34
        if "ai_text_testing_guide" in lowered_path or "human_data_collection_guide" in lowered_path:
            boost -= 0.22

    if any(
        token in lowered_question
        for token in ("更新的预训练", "更强模型", "为什么不换", "roberta", "不用更新")
    ):
        if "docs/project/advisor_academic_qa.md" in lowered_path:
            boost += 0.34
        if "docs/project/defense_kb_curated.md" in lowered_path:
            boost += 0.28
        if "英文" in lowered_question and "docs/project/advisor_academic_qa.md" in lowered_path:
            boost += 0.08

    if any(
        token in lowered_question
        for token in ("模型大小", "模型多大", "显存", "训练时长", "gpu", "部署效率", "推理速度")
    ):
        if "docs/project/defense_kb_curated.md" in lowered_path:
            boost += 0.28
        if "docs/project/advisor_academic_qa.md" in lowered_path:
            boost += 0.24
        if "docs/thesis/chapter5_experiments_filled.md" in lowered_path:
            boost += 0.34
        if "docs/thesis/thesis_data_reference.md" in lowered_path:
            boost += 0.3
        if "docs/deployment.md" in lowered_path or "docker" in lowered_path:
            boost -= 0.28

    if any(
        token in lowered_question
        for token in (
            "别的方法",
            "基线方法",
            "对比方法",
            "fasttext",
            "textcnn",
            "dpcnn",
            "bert-bigru",
        )
    ):
        if "docs/project/defense_kb_curated.md" in lowered_path:
            boost += 0.34
        if "docs/project/advisor_academic_qa.md" in lowered_path:
            boost += 0.3
        if "docs/thesis/chapter5_experiments_filled.md" in lowered_path:
            boost += 0.24

    if any(token in lowered_question for token in ("英文文本", "英文", "英语", "多语")):
        if "docs/project/advisor_academic_qa.md" in lowered_path:
            boost += 0.28
        if "docs/project/defense_kb_curated.md" in lowered_path:
            boost += 0.18

    if any(
        token in lowered_question
        for token in ("数据治理", "数据清洗", "训练集", "数据集", "样本来源", "来源覆盖")
    ):
        if "docs/project/defense_kb_curated.md" in lowered_path:
            boost += 0.28
        if "docs/project/advisor_academic_qa.md" in lowered_path:
            boost += 0.24
        if "project_technical_deep_dive" in lowered_path:
            boost += 0.22
        if "docs/project/defense_current_status.md" in lowered_path:
            boost += 0.16

    if any(
        token in lowered_question
        for token in (
            "创新",
            "亮点",
            "贡献",
            "[sep]",
            "边界",
            "边界定位",
            "混合文本",
            "bert",
            "gpt",
            "llama",
            "bert-bigru",
            "基线",
        )
    ):
        if "docs/project/defense_kb_curated.md" in lowered_path:
            boost += 0.26
        if "docs/project/advisor_academic_qa.md" in lowered_path:
            boost += 0.24
        if "project_technical_deep_dive" in lowered_path:
            boost += 0.24
        if lowered_path == "readme.md":
            boost += 0.18
        if "docs/project/defense_current_status.md" in lowered_path:
            boost += 0.12

    if any(
        token in lowered_question for token in ("局限", "风险", "不足", "过拟合", "泛化", "质疑")
    ):
        if "docs/project/defense_kb_curated.md" in lowered_path:
            boost += 0.24
        if "docs/project/advisor_academic_qa.md" in lowered_path:
            boost += 0.22
        if "docs/project/defense_current_status.md" in lowered_path:
            boost += 0.22
        if "project_technical_deep_dive" in lowered_path:
            boost += 0.1

    if any(
        token in lowered_question
        for token in (
            "研究问题",
            "研究目标",
            "应用价值",
            "temperature scaling",
            "双层检测架构",
            "创新点",
            "应用场景",
            "工程价值",
            "分布外",
            "新模型",
            "局限性",
            "混淆矩阵",
            "召回率",
            "推理速度",
            "部署",
        )
    ):
        if "docs/project/defense_kb_curated.md" in lowered_path:
            boost += 0.34
        if "docs/project/advisor_academic_qa.md" in lowered_path:
            boost += 0.28

    return boost


# ---------------------------------------------------------------------------
# Dense retrieval via OpenAI Embedding API
# ---------------------------------------------------------------------------

_DENSE_CACHE_TTL = 3600.0  # 1 hour
_dense_embedding_cache: dict[str, tuple[list[float], float]] = {}


def _get_openai_embedding_config() -> tuple[str, str, str] | None:
    """Return (api_key, api_base, model) for embedding calls, or None if unavailable."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    api_base = os.getenv("OPENAI_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
    model = os.getenv("DC_EMBEDDING_MODEL", "text-embedding-3-small")
    return api_key, api_base, model


async def _get_embeddings_async(texts: list[str]) -> list[list[float]] | None:
    """Get embeddings for a list of texts via OpenAI Embedding API."""
    config = _get_openai_embedding_config()
    if not config:
        return None
    api_key, api_base, model = config
    try:
        import httpx

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{api_base}/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "input": texts},
            )
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", [])
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
    except Exception:
        return None


def _get_embeddings_sync(texts: list[str]) -> list[list[float]] | None:
    """Synchronous embedding retrieval with in-memory cache."""
    now = time.time()
    uncached_texts = []
    uncached_indices = []
    results: list[list[float] | None] = [None] * len(texts)

    for i, text in enumerate(texts):
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        cached = _dense_embedding_cache.get(key)
        if cached and (now - cached[1]) < _DENSE_CACHE_TTL:
            results[i] = cached[0]
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)

    if uncached_texts:
        config = _get_openai_embedding_config()
        if not config:
            return None
        api_key, api_base, model = config
        try:
            import httpx

            resp = httpx.post(
                f"{api_base}/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "input": uncached_texts},
                timeout=15,
            )
            if resp.status_code != 200:
                return None
            data = resp.json().get("data", [])
            embeddings = [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
            for idx, emb in zip(uncached_indices, embeddings):
                results[idx] = emb
                key = hashlib.md5(texts[idx].encode("utf-8")).hexdigest()
                _dense_embedding_cache[key] = (emb, now)
        except Exception:
            return None

    return [r for r in results if r is not None] if all(r is not None for r in results) else None


def _cosine_similarity_dense(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two dense vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

_RRF_K = 60  # standard RRF constant


def _rrf_fusion(
    sparse_ranked: list[tuple[float, int]],
    dense_ranked: list[tuple[float, int]],
    top_k: int = 10,
) -> list[tuple[float, int]]:
    """Fuse two ranked lists using Reciprocal Rank Fusion."""
    rrf_scores: dict[int, float] = {}

    for rank, (_, idx) in enumerate(sparse_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank + 1)

    for rank, (_, idx) in enumerate(dense_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank + 1)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, idx) for idx, score in fused[:top_k]]


# ---------------------------------------------------------------------------
# QA v2 JSON loader
# ---------------------------------------------------------------------------

_QA_V2_CACHE: list[dict[str, Any]] | None = None


def load_qa_v2() -> list[dict[str, Any]]:
    """Load DEFENSE_KB_QA_v2.json from docs/project/."""
    global _QA_V2_CACHE
    if _QA_V2_CACHE is not None:
        return _QA_V2_CACHE

    qa_path = PROJECT_ROOT / "docs" / "project" / "DEFENSE_KB_QA_v2.json"
    if not qa_path.exists():
        _QA_V2_CACHE = []
        return _QA_V2_CACHE

    try:
        with open(qa_path, "r", encoding="utf-8") as f:
            _QA_V2_CACHE = json.load(f)
    except (json.JSONDecodeError, OSError):
        _QA_V2_CACHE = []

    return _QA_V2_CACHE


def search_qa_v2(question: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Search QA v2 by keyword matching. Returns top_k matching Q&A pairs."""
    qa_pairs = load_qa_v2()
    if not qa_pairs:
        return []

    lowered = question.lower()
    scored: list[tuple[float, dict[str, Any]]] = []

    for qa in qa_pairs:
        keywords = qa.get("keywords", [])
        if not keywords:
            continue
        hits = sum(1 for kw in keywords if kw.lower() in lowered)
        if hits > 0:
            scored.append((hits / len(keywords), qa))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [qa for _, qa in scored[:top_k]]


# ---------------------------------------------------------------------------
# Multi-turn query rewriting
# ---------------------------------------------------------------------------

_REFERENTIAL_PATTERNS = re.compile(
    r"^(它|这个|那个|那|这|上述|刚才|之前|上面|你刚|你说|你提|为什么|怎么|如何|具体|详细|展开|补充|还有|然后|接着|继续|哪些|什么)"
)


def rewrite_query_with_history(
    question: str,
    history: Sequence[dict[str, str]] | None = None,
) -> str:
    """Expand elliptical follow-up questions with recent conversation context.

    If the question looks referential (short + contains pronouns/demonstratives
    + has history), append recent context to make it self-contained.
    """
    if not history:
        return question

    lowered = question.strip().lower()
    # Trigger if: short question + (has pronouns/demonstratives OR has history)
    is_referential = (
        len(question.strip()) < 20
        and _REFERENTIAL_PATTERNS.search(question.strip())
    ) or (
        len(question.strip()) < 12
        and history
    )

    if not is_referential:
        return question

    recent_lines: list[str] = []
    for msg in list(history)[-4:]:
        content = re.sub(r"\s+", " ", (msg.get("content") or "").strip())
        if not content:
            continue
        if len(content) > 120:
            content = content[:117].rstrip() + "..."
        recent_lines.append(content)

    if not recent_lines:
        return question

    context_block = " / ".join(recent_lines)
    return f"{question}\n最近对话上下文：{context_block}"


class ProjectKnowledgeIndex:
    """In-memory retrieval index for project-defense Q&A."""

    def __init__(
        self,
        root: Path = PROJECT_ROOT,
        patterns: Sequence[str] = DEFAULT_SOURCE_PATTERNS,
    ) -> None:
        self.root = root
        self.patterns = patterns
        self.chunks: list[KnowledgeChunk] = []
        self.doc_freq: dict[str, int] = {}
        self.chunk_vectors: list[dict[str, float]] = []
        self.chunk_norms: list[float] = []
        self.chunk_dense_embeddings: list[list[float]] = []
        self.dense_available: bool = False
        self.code_symbol_index: dict[str, list[CodeSymbol]] = {}

    @property
    def index_ready(self) -> bool:
        return (
            bool(self.chunks)
            and bool(self.chunk_vectors)
            and len(self.chunk_vectors) == len(self.chunks)
        )

    @property
    def source_count(self) -> int:
        return len({chunk.path for chunk in self.chunks})

    def refresh(self) -> None:
        """Rebuild the retrieval index from disk."""
        self.chunks = build_project_chunks(root=self.root, patterns=self.patterns)
        self.doc_freq = {}
        self.chunk_vectors = []
        self.chunk_norms = []
        self.chunk_dense_embeddings = []
        self.dense_available = False
        self.code_symbol_index = build_code_symbol_index(root=self.root)
        if not self.chunks:
            return

        chunk_counts = [
            _char_ngram_counts(_build_chunk_search_text(chunk)) for chunk in self.chunks
        ]
        for counts in chunk_counts:
            for token in counts:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        doc_count = len(self.chunks)
        for counts in chunk_counts:
            vector, norm = _build_sparse_tfidf_vector(
                counts,
                doc_freq=self.doc_freq,
                doc_count=doc_count,
            )
            self.chunk_vectors.append(vector)
            self.chunk_norms.append(norm)

        # Build dense embedding index (best effort)
        if os.getenv("DC_DISABLE_DENSE", "0").strip().lower() not in {"1", "true", "yes"}:
            chunk_texts = [_build_chunk_search_text(chunk)[:512] for chunk in self.chunks]
            embeddings = _get_embeddings_sync(chunk_texts)
            if embeddings and len(embeddings) == len(self.chunks):
                self.chunk_dense_embeddings = embeddings
                self.dense_available = True

    def search(
        self,
        question: str,
        top_k: int = 5,
        min_score: float | None = PROJECT_QA_MIN_RELEVANCE_SCORE,
    ) -> list[KnowledgeHit]:
        """Return the most relevant local evidence using hybrid sparse+dense retrieval."""
        if not question.strip():
            return []
        if build_project_decline_answer(question):
            return []

        if not self.index_ready:
            self.refresh()
        if not self.index_ready:
            return []

        retrieval_query = _expand_project_query(question)
        top_n = max(top_k * 3, 20)

        # --- Sparse retrieval (char n-gram TF-IDF) ---
        sparse_scores = self._sparse_search(retrieval_query)
        sparse_ranked = sorted(
            enumerate(sparse_scores), key=lambda x: x[1], reverse=True
        )[:top_n]

        # --- Dense retrieval (embedding API) ---
        if self.dense_available and self.chunk_dense_embeddings:
            query_emb = _get_embeddings_sync([retrieval_query[:512]])
            if query_emb:
                dense_scores = [
                    _cosine_similarity_dense(query_emb[0], chunk_emb)
                    for chunk_emb in self.chunk_dense_embeddings
                ]
                dense_ranked = sorted(
                    enumerate(dense_scores), key=lambda x: x[1], reverse=True
                )[:top_n]
                # RRF fusion
                fused = _rrf_fusion(
                    [(s, i) for i, s in sparse_ranked],
                    [(s, i) for i, s in dense_ranked],
                    top_k=top_n,
                )
                # Use RRF rank scores, but keep sparse scores for relevance gating
                ranked_indices = [idx for _, idx in fused]
                rrf_lookup = {idx: score for score, idx in fused}
            else:
                ranked_indices = [i for i, _ in sparse_ranked]
                rrf_lookup = {}
        else:
            ranked_indices = [i for i, _ in sparse_ranked]
            rrf_lookup = {}

        # --- Apply keyword/title/section/path boosts ---
        keywords = _extract_question_keywords(retrieval_query)
        adjusted_scores: list[tuple[float, int]] = []
        for index in ranked_indices:
            base_score = rrf_lookup.get(index, sparse_scores[index])
            chunk = self.chunks[index]
            search_text_lower = _build_chunk_search_text(chunk).lower()
            keyword_bonus = sum(0.02 for keyword in keywords if keyword in search_text_lower)
            title_bonus = sum(0.03 for keyword in keywords if keyword in chunk.title.lower())
            section_bonus = sum(0.04 for keyword in keywords if keyword in chunk.section.lower())
            path_boost = _source_path_boost(chunk.path) + _question_path_boost(
                retrieval_query,
                chunk.path,
            )
            relevance_gate = min(1.0, max(0.0, float(base_score) * 6))
            adjusted = (
                float(base_score)
                + keyword_bonus
                + title_bonus
                + section_bonus
                + path_boost * relevance_gate
            )
            adjusted_scores.append((adjusted, index))

        adjusted_lookup = {index: adjusted for adjusted, index in adjusted_scores}
        ranked_indices = [index for _, index in sorted(adjusted_scores, reverse=True)]
        hits: list[KnowledgeHit] = []
        seen_paths: set[str] = set()

        for chunk_index in ranked_indices:
            adjusted_score = adjusted_lookup[chunk_index]
            raw_score = sparse_scores[chunk_index]
            if raw_score <= 0 or (min_score is not None and adjusted_score < min_score):
                continue

            chunk = self.chunks[int(chunk_index)]
            # Prefer source diversity so the answer cites multiple files.
            if chunk.path in seen_paths and len(hits) >= top_k:
                continue

            hits.append(
                KnowledgeHit(
                    chunk=chunk,
                    score=adjusted_score,
                    excerpt=_build_excerpt(chunk.content),
                )
            )
            seen_paths.add(chunk.path)
            if len(hits) >= top_k:
                break

        return hits

    def _sparse_search(self, query: str) -> list[float]:
        """Compute sparse TF-IDF scores for all chunks against the query."""
        query_counts = _char_ngram_counts(query)
        query_vector, query_norm = _build_sparse_tfidf_vector(
            query_counts,
            doc_freq=self.doc_freq,
            doc_count=len(self.chunks),
        )
        if not query_vector or query_norm <= 0.0:
            return [0.0] * len(self.chunks)

        return [
            _cosine_similarity_sparse(
                query_vector,
                chunk_vector,
                left_norm=query_norm,
                right_norm=self.chunk_norms[index],
            )
            for index, chunk_vector in enumerate(self.chunk_vectors)
        ]

    def resolve_code_symbol(self, term: str) -> CodeSymbol | None:
        """Resolve a local code symbol by exact then fuzzy match."""
        normalized = term.strip().casefold()
        if not normalized:
            return None

        exact = self.code_symbol_index.get(normalized)
        if exact:
            return exact[0]

        for symbol_key, symbols in self.code_symbol_index.items():
            if normalized in symbol_key or symbol_key in normalized:
                return symbols[0]

        return None


def build_extractive_answer(question: str, hits: Sequence[KnowledgeHit]) -> str:
    """Generate a grounded fallback answer without relying on an LLM.

    v2: Uses QA v2 JSON for keyword matches, then falls back to sentence-level
    extraction from retrieved hits. No more hardcoded if-else templates.
    """
    decline_answer = build_project_decline_answer(question)
    if decline_answer:
        return decline_answer

    # 1. Try QA v2 exact match first
    qa_matches = search_qa_v2(question, top_k=1)
    if qa_matches and qa_matches[0].get("answer"):
        return f"可以先这样回答：{qa_matches[0]['answer']}"

    # 2. Fall back to hit-grounded sentence extraction
    if not hits:
        return "我暂时没有在当前仓库里检索到足够相关的资料，建议先换一种更具体的问法。"

    source_labels = {hit.chunk.chunk_id: index + 1 for index, hit in enumerate(hits)}
    keywords = _extract_question_keywords(question)
    scored_sentences: list[tuple[float, str, int]] = []

    for hit in hits:
        sentences = [
            part.strip()
            for part in _SENTENCE_SPLIT_PATTERN.split(hit.chunk.content)
            if part.strip()
        ]
        for sentence in sentences:
            if _looks_like_low_value_sentence(sentence):
                continue
            lowered = sentence.lower()
            keyword_bonus = sum(1 for keyword in keywords if keyword in lowered)
            if keywords and keyword_bonus == 0:
                continue
            score = hit.score + keyword_bonus * 0.08
            scored_sentences.append((score, sentence, source_labels[hit.chunk.chunk_id]))

    if not scored_sentences:
        top_hit = hits[0]
        return (
            "根据仓库里最相关的资料，当前最接近问题的证据来自 "
            f"`{top_hit.chunk.path}`：{top_hit.excerpt}"
        )

    selected_lines: list[str] = []
    seen_sentences: set[str] = set()
    for _, sentence, source_label in sorted(
        scored_sentences, key=lambda item: item[0], reverse=True
    ):
        normalized = re.sub(r"\s+", " ", sentence)
        if normalized in seen_sentences:
            continue
        seen_sentences.add(normalized)
        selected_lines.append(f"{normalized} [{source_label}]")
        if len(selected_lines) == 4:
            break

    if not selected_lines:
        top_hit = hits[0]
        return (
            "根据仓库里最相关的资料，当前最接近问题的证据来自 "
            f"`{top_hit.chunk.path}`：{top_hit.excerpt}"
        )

    return "根据仓库当前资料，可以先这样回答：\n" + "\n".join(
        f"{index}. {line}" for index, line in enumerate(selected_lines, start=1)
    )
