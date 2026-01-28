#!/usr/bin/env python3
"""
Clean prompt residue from generated AI texts.

Some models (e.g., glm-4.7, gpt-4) output analysis/planning content before
the actual text. This script detects and removes such residue patterns.

Patterns handled:
- "1. **分析请求：**" followed by analysis
- "2. **起草 - 逐节进行：**" followed by actual content
- Structured section markers like "**定义：**", "**对比：**"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Pattern: Text starts with "1. **分析请求" or similar
ANALYSIS_PATTERN = re.compile(
    r"^[\d\*\.\s]*分析请求[\s\S]*?(?=\n*[\d\*\.\s]*起草|$)",
    re.MULTILINE
)

# Pattern: "2. **起草 - 逐节进行：**" section header
DRAFT_HEADER_PATTERN = re.compile(
    r"[\d\*\.\s]*起草[^：:]*[：:]\s*\n*",
    re.MULTILINE
)

# Pattern: Section markers like "**定义：**", "*   **对比（与SGD）：**"
SECTION_MARKER_PATTERN = re.compile(
    r"^\s*[\*\-\•]+\s*\*{0,2}([^：:*]+)[：:]\s*\*{0,2}\s*",
    re.MULTILINE
)

# Pattern: Markdown bold markers
BOLD_PATTERN = re.compile(r"\*\*([^*]+)\*\*")

# Pattern: Excessive whitespace
EXCESSIVE_WHITESPACE = re.compile(r"\n{3,}")


def detect_residue_type(text: str) -> Optional[str]:
    """Detect type of prompt residue in text."""
    text_start = text[:200].lower()

    if "分析请求" in text_start:
        return "analysis_pattern"
    if re.match(r"^\d+\.\s*\*{0,2}", text):
        # Starts with numbered list that looks like planning
        if any(keyword in text_start for keyword in ["角色", "输出", "限制", "要求"]):
            return "planning_pattern"
    return None


def extract_clean_content(text: str, residue_type: str) -> Tuple[str, bool]:
    """Extract clean content from text with residue.

    Strategy: Find the actual content section after the planning/analysis parts.
    Look for patterns like "起草" or actual content paragraphs.

    Returns:
        Tuple of (cleaned_text, was_cleaned)
    """
    original = text

    if residue_type == "analysis_pattern":
        # Find where actual content starts
        # Pattern 1: After "起草" section header, find actual prose
        draft_match = re.search(r"起草[^：:]*[：:]\s*\n*", text)
        if draft_match:
            # Content is after draft header
            text = text[draft_match.end():]

        # Now extract actual content from structured sections
        # Pattern: "**定义：** content" or "*   **XXX：** content"
        content_parts = []
        section_pattern = re.compile(
            r"[\*\s]*\*{2}([^*：:]+)[：:]\*{2}\s*(.+?)(?=\n[\*\s]*\*{2}[^*]+[：:]\*{2}|$)",
            re.DOTALL
        )
        matches = section_pattern.findall(text)

        if matches:
            # Extract content from each section
            for section_name, content in matches:
                # Skip meta sections like "角色", "输出", "限制" etc
                if any(skip in section_name for skip in ["角色", "输出", "限制", "要求", "加分"]):
                    continue
                content = content.strip()
                if content and len(content) > 20:
                    # Remove section name if it appears in content
                    content = re.sub(r"^" + re.escape(section_name) + r"[：:]\s*", "", content)
                    content_parts.append(content)

            if content_parts:
                text = "\n\n".join(content_parts)
            else:
                # Fallback: try to find prose paragraphs
                text = _extract_prose_fallback(text)
        else:
            # No structured sections found, try prose extraction
            text = _extract_prose_fallback(text)

    # Final cleanup
    # Remove markdown bold markers
    text = BOLD_PATTERN.sub(r"\1", text)
    # Remove bullet points at start of lines
    text = re.sub(r"^\s*[\*\-\•·]\s*", "", text, flags=re.MULTILINE)
    # Clean excessive whitespace
    text = EXCESSIVE_WHITESPACE.sub("\n\n", text)
    text = text.strip()

    was_cleaned = text != original
    return text, was_cleaned


def _extract_prose_fallback(text: str) -> str:
    """Fallback: extract prose-like paragraphs from text."""
    lines = text.split("\n")
    prose_lines = []
    in_content = False

    for line in lines:
        line = line.strip()
        if not line:
            if in_content:
                prose_lines.append("")
            continue

        # Skip lines that look like meta/planning
        skip_patterns = [
            r"^[\d\.\*\s]*分析",
            r"^[\d\.\*\s]*起草",
            r"^\*{0,2}(角色|输出|限制|要求|加分|字数|主题)[：:]",
            r"^\d+\.\s*\*{2}",  # Numbered bold headers
        ]
        if any(re.match(p, line, re.IGNORECASE) for p in skip_patterns):
            continue

        # Check if line looks like actual content (has Chinese prose)
        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", line))
        if chinese_count > 10 or (chinese_count > 5 and len(line) > 20):
            in_content = True
            # Clean the line
            line = re.sub(r"^\s*[\*\-\•·]+\s*", "", line)
            line = re.sub(r"^\s*\*{2}[^*]+[：:]\*{2}\s*", "", line)
            prose_lines.append(line)

    return "\n".join(prose_lines).strip()


def process_record(record: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Process a single record.

    Returns:
        Tuple of (processed_record, status) where status is:
        - "cleaned": residue was detected and removed
        - "clean": no residue detected
        - "skipped": cleaning failed or empty result
    """
    text = record.get("text", "")
    if not text:
        return record, "skipped"

    residue_type = detect_residue_type(text)
    if not residue_type:
        return record, "clean"

    cleaned_text, was_cleaned = extract_clean_content(text, residue_type)

    if not cleaned_text or len(cleaned_text) < 50:
        # Cleaning failed or result too short
        return record, "skipped"

    if was_cleaned:
        new_record = record.copy()
        new_record["text"] = cleaned_text
        new_record["cleaned_residue"] = 1
        new_record["original_residue_type"] = residue_type
        return new_record, "cleaned"

    return record, "clean"


def process_file(
    input_path: Path,
    output_path: Path,
    skipped_path: Optional[Path] = None
) -> Dict[str, int]:
    """Process a JSONL file and clean prompt residue.

    Returns:
        Statistics dict with counts
    """
    stats = {"total": 0, "clean": 0, "cleaned": 0, "skipped": 0}

    with input_path.open("r", encoding="utf-8") as fin:
        with output_path.open("w", encoding="utf-8") as fout:
            skipped_fout = None
            if skipped_path:
                skipped_fout = skipped_path.open("w", encoding="utf-8")

            try:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    stats["total"] += 1
                    processed_record, status = process_record(record)
                    stats[status] += 1

                    if status == "skipped":
                        if skipped_fout:
                            skipped_fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    else:
                        fout.write(json.dumps(processed_record, ensure_ascii=False) + "\n")
            finally:
                if skipped_fout:
                    skipped_fout.close()

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean prompt residue from generated AI texts."
    )
    parser.add_argument(
        "input",
        help="Input JSONL file or directory containing JSONL files",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSONL file or directory (default: input_cleaned.jsonl)",
    )
    parser.add_argument(
        "--skipped",
        help="File to save skipped records (default: none)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only analyze and report, don't write output",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        # Process all JSONL files in directory
        jsonl_files = sorted(input_path.glob("*.jsonl"))
        if not jsonl_files:
            print(f"No JSONL files found in {input_path}", file=sys.stderr)
            return 1

        total_stats = {"total": 0, "clean": 0, "cleaned": 0, "skipped": 0}

        for jsonl_file in jsonl_files:
            if "_cleaned" in jsonl_file.name or "_skipped" in jsonl_file.name:
                continue

            output_file = jsonl_file.with_name(
                jsonl_file.stem + "_cleaned" + jsonl_file.suffix
            )
            skipped_file = None
            if args.skipped:
                skipped_file = jsonl_file.with_name(
                    jsonl_file.stem + "_cleaning_skipped" + jsonl_file.suffix
                )

            if args.dry_run:
                # Just analyze
                with jsonl_file.open("r", encoding="utf-8") as f:
                    residue_count = 0
                    for line in f:
                        if line.strip():
                            try:
                                record = json.loads(line)
                                if detect_residue_type(record.get("text", "")):
                                    residue_count += 1
                                total_stats["total"] += 1
                            except:
                                pass
                    total_stats["cleaned"] += residue_count
                print(f"[dry-run] {jsonl_file.name}: residue={residue_count}")
            else:
                stats = process_file(jsonl_file, output_file, skipped_file)
                for k, v in stats.items():
                    total_stats[k] += v
                print(
                    f"[processed] {jsonl_file.name}: "
                    f"total={stats['total']} cleaned={stats['cleaned']} "
                    f"skipped={stats['skipped']}"
                )

        print()
        print(f"=== Total Stats ===")
        print(f"Total records: {total_stats['total']}")
        print(f"Clean (no residue): {total_stats['clean']}")
        print(f"Cleaned (residue removed): {total_stats['cleaned']}")
        print(f"Skipped (cleaning failed): {total_stats['skipped']}")
    else:
        # Process single file
        if not input_path.exists():
            print(f"Input file not found: {input_path}", file=sys.stderr)
            return 1

        output_path = Path(args.output) if args.output else input_path.with_name(
            input_path.stem + "_cleaned" + input_path.suffix
        )
        skipped_path = Path(args.skipped) if args.skipped else None

        if args.dry_run:
            residue_count = 0
            total = 0
            with input_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            if detect_residue_type(record.get("text", "")):
                                residue_count += 1
                            total += 1
                        except:
                            pass
            print(f"[dry-run] {input_path.name}: total={total} residue={residue_count}")
        else:
            stats = process_file(input_path, output_path, skipped_path)
            print(f"Processed: {input_path}")
            print(f"Output: {output_path}")
            print(f"Stats: total={stats['total']} clean={stats['clean']} "
                  f"cleaned={stats['cleaned']} skipped={stats['skipped']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
