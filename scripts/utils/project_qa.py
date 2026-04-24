"""Lightweight project-knowledge retrieval for thesis defense Q&A.

This module builds a small in-process retrieval index from the current
repository so the API can answer questions about the project using local
documents and selected code files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree
from zipfile import ZipFile

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency in lean envs
    PdfReader = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scripts.utils.paths import PATHS, PROJECT_ROOT

DEFAULT_SOURCE_PATTERNS = (
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
    "models/*/training_log.json",
    "datasets/project_qa_uploads/**/*",
)

SUPPORTED_SOURCE_SUFFIXES = {".md", ".txt", ".json", ".py", ".docx", ".pdf", ".pptx"}

_QUESTION_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?；;])\s+|\n+")
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


@dataclass(slots=True)
class KnowledgeChunk:
    """A single retrievable chunk from the repository."""

    chunk_id: str
    path: str
    title: str
    content: str


@dataclass(slots=True)
class KnowledgeHit:
    """A scored retrieval hit."""

    chunk: KnowledgeChunk
    score: float
    excerpt: str


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
                name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")
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
    return [path for path in sorted(upload_dir.rglob("*")) if path.is_file() and path.suffix.lower() in SUPPORTED_SOURCE_SUFFIXES]


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
        for index, chunk_text in enumerate(chunk_project_text(source_text), start=1):
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{relative_path}#{index}",
                    path=relative_path,
                    title=source_path.stem,
                    content=chunk_text,
                )
            )

    return chunks


def _build_excerpt(text: str, *, limit: int = 240) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _extract_question_keywords(question: str) -> list[str]:
    keywords: list[str] = []
    for token in _QUESTION_TOKEN_PATTERN.findall(question):
        lowered = token.lower()
        if lowered in _STOPWORDS:
            continue
        keywords.append(lowered)
    return keywords


def _looks_like_low_value_sentence(sentence: str) -> bool:
    normalized = sentence.strip()
    if len(normalized) < 8:
        return True

    lowered = normalized.lower()
    if any(token in lowered for token in ('"question":', '"answer":', 'def ', 'class ', 'import ')):
        return True

    punctuation_count = sum(1 for char in normalized if char in "{}[]()=:_/\\\"'")
    if punctuation_count / max(len(normalized), 1) > 0.18 and not re.search(r"[\u4e00-\u9fff]", normalized):
        return True

    return False


def _source_path_boost(path: str) -> float:
    lowered = path.lower()
    boost = 0.0

    if "docs/project/defense_current_status.md" in lowered:
        boost += 0.18
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
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    @property
    def source_count(self) -> int:
        return len({chunk.path for chunk in self.chunks})

    def refresh(self) -> None:
        """Rebuild the retrieval index from disk."""
        self.chunks = build_project_chunks(root=self.root, patterns=self.patterns)
        if not self.chunks:
            self.vectorizer = None
            self.matrix = None
            return

        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([chunk.content for chunk in self.chunks])

    def search(self, question: str, top_k: int = 5) -> list[KnowledgeHit]:
        """Return the most relevant local evidence for a project question."""
        if not question.strip():
            return []

        if self.vectorizer is None or self.matrix is None:
            self.refresh()
        if self.vectorizer is None or self.matrix is None:
            return []

        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        keywords = _extract_question_keywords(question)
        adjusted_scores: list[tuple[float, int]] = []
        for index, score in enumerate(scores):
            chunk = self.chunks[index]
            content_lower = chunk.content.lower()
            keyword_bonus = sum(0.02 for keyword in keywords if keyword in content_lower)
            path_boost = _source_path_boost(chunk.path)
            relevance_gate = min(1.0, max(0.0, float(score) * 6))
            adjusted = float(score) + keyword_bonus + path_boost * relevance_gate
            adjusted_scores.append((adjusted, index))

        ranked_indices = [index for _, index in sorted(adjusted_scores, reverse=True)]
        hits: list[KnowledgeHit] = []
        seen_paths: set[str] = set()

        for chunk_index in ranked_indices:
            score = float(scores[chunk_index])
            if score <= 0:
                continue

            chunk = self.chunks[int(chunk_index)]
            # Prefer source diversity so the answer cites multiple files.
            if chunk.path in seen_paths and len(hits) >= top_k:
                continue

            hits.append(KnowledgeHit(chunk=chunk, score=score, excerpt=_build_excerpt(chunk.content)))
            seen_paths.add(chunk.path)
            if len(hits) >= top_k:
                break

        return hits


def build_extractive_answer(question: str, hits: Sequence[KnowledgeHit]) -> str:
    """Generate a grounded fallback answer without relying on an LLM."""
    if not hits:
        return "我暂时没有在当前仓库里检索到足够相关的资料，建议先换一种更具体的问法。"

    source_labels = {hit.chunk.chunk_id: index + 1 for index, hit in enumerate(hits)}
    keywords = _extract_question_keywords(question)
    scored_sentences: list[tuple[float, str, int]] = []

    for hit in hits:
        sentences = [part.strip() for part in _SENTENCE_SPLIT_PATTERN.split(hit.chunk.content) if part.strip()]
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
    for _, sentence, source_label in sorted(scored_sentences, key=lambda item: item[0], reverse=True):
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
