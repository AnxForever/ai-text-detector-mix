"""Tests for the lightweight project QA retrieval module."""

from __future__ import annotations

from zipfile import ZipFile

from scripts.utils.project_qa import (
    ProjectKnowledgeIndex,
    build_extractive_answer,
    read_project_source,
)


def test_project_qa_index_retrieves_relevant_defense_doc(tmp_path):
    (tmp_path / "docs" / "project").mkdir(parents=True)
    (tmp_path / "api").mkdir(parents=True)

    (tmp_path / "README.md").write_text("项目总览：这是一个中文AI文本检测系统。", encoding="utf-8")
    (tmp_path / "api" / "api.py").write_text(
        "MODEL_VERSION = 'bert_v11c_boundary_fix'\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "project" / "DEFENSE_CURRENT_STATUS.md").write_text(
        "三集平均准确率 98.56%，独立评估集准确率 98.57%，当前推荐模型为 bert_v11c_boundary_fix。",
        encoding="utf-8",
    )

    index = ProjectKnowledgeIndex(
        root=tmp_path,
        patterns=("README.md", "docs/project/*.md", "api/api.py"),
    )
    index.refresh()

    hits = index.search("当前三集平均准确率是多少", top_k=3)

    assert hits
    assert hits[0].chunk.path == "docs/project/DEFENSE_CURRENT_STATUS.md"
    assert index.source_count == 3


def test_build_extractive_answer_uses_retrieved_evidence(tmp_path):
    (tmp_path / "docs" / "project").mkdir(parents=True)

    (tmp_path / "docs" / "project" / "FINAL_RESULTS.md").write_text(
        "本项目采用双层检测架构：第一层分类器负责 Human/AI/Mixed 判定，"
        "第二层边界检测器负责定位AI续写起点。",
        encoding="utf-8",
    )

    index = ProjectKnowledgeIndex(root=tmp_path, patterns=("docs/project/*.md",))
    hits = index.search("项目为什么是双层检测架构", top_k=2)
    answer = build_extractive_answer("项目为什么是双层检测架构", hits)

    assert "双层检测架构" in answer
    assert "边界检测器" in answer


def test_project_qa_reads_json_metrics_as_searchable_text(tmp_path):
    (tmp_path / "models" / "bert_v11c_boundary_fix").mkdir(parents=True)

    metrics_path = tmp_path / "models" / "bert_v11c_boundary_fix" / "eval_comparison.json"
    metrics_path.write_text(
        '{"bert_v11c_boundary_fix": {"three_set_avg": 98.56, "independent_data": {"accuracy": 98.57}}}',
        encoding="utf-8",
    )

    content = read_project_source(metrics_path)
    assert "three_set_avg: 98.56" in content
    assert "independent_data.accuracy: 98.57" in content

    index = ProjectKnowledgeIndex(root=tmp_path, patterns=("models/*/eval_comparison.json",))
    index.refresh()
    hits = index.search("三集平均准确率 98.56", top_k=1)

    assert hits
    assert hits[0].chunk.path == "models/bert_v11c_boundary_fix/eval_comparison.json"


def test_project_qa_reads_docx_as_searchable_text(tmp_path):
    proposal_path = tmp_path / "开题报告.docx"
    document_xml = """
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:r><w:t>基于BERT微调的中文AI生成文本检测</w:t></w:r></w:p>
        <w:p><w:r><w:t>研究目标包括分类检测与边界定位。</w:t></w:r></w:p>
      </w:body>
    </w:document>
    """.strip()

    with ZipFile(proposal_path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    content = read_project_source(proposal_path)
    assert "中文AI生成文本检测" in content
    assert "边界定位" in content

    index = ProjectKnowledgeIndex(root=tmp_path, patterns=("*.docx",))
    index.refresh()
    hits = index.search("边界定位是什么", top_k=1)

    assert hits
    assert hits[0].chunk.path == "开题报告.docx"
