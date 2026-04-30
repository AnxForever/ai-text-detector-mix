"""Tests for the lightweight project QA retrieval module."""

from __future__ import annotations

from zipfile import ZipFile

from scripts.utils.project_qa import (
    PROJECT_ROOT,
    ProjectKnowledgeIndex,
    build_code_symbol_index,
    build_contextual_retrieval_query,
    build_extractive_answer,
    discover_project_sources,
    read_project_source,
)


def test_project_qa_index_prioritizes_academic_qa_doc(tmp_path):
    (tmp_path / "docs" / "project").mkdir(parents=True)

    (tmp_path / "docs" / "project" / "ADVISOR_ACADEMIC_QA.md").write_text(
        "为什么本文选择 BERT 而不是 GPT / LLaMA 一类生成模型？\n"
        "因为该任务本质上是判别任务，BERT 更适合做监督分类。\n\n"
        "Temperature Scaling 与 ECE 在本文中分别说明什么？\n"
        "Temperature Scaling 用于后置概率校准，ECE 用于衡量置信度偏差。\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "project" / "DEFENSE_CURRENT_STATUS.md").write_text(
        "当前推荐模型为 bert_v11c_boundary_fix，ECE 为 0.0034。",
        encoding="utf-8",
    )

    index = ProjectKnowledgeIndex(root=tmp_path, patterns=("docs/project/*.md",))
    index.refresh()

    bert_hits = index.search("为什么本文选择 BERT 而不是 GPT / LLaMA 一类生成模型？", top_k=2)
    ece_hits = index.search("Temperature Scaling 与 ECE 在本文中分别说明什么？", top_k=2)

    assert bert_hits
    assert ece_hits
    assert bert_hits[0].chunk.path == "docs/project/ADVISOR_ACADEMIC_QA.md"
    assert ece_hits[0].chunk.path == "docs/project/ADVISOR_ACADEMIC_QA.md"


def test_build_extractive_answer_covers_academic_recommended_questions():
    bert_answer = build_extractive_answer(
        "为什么本文选择 BERT 作为主模型，而不是 GPT / LLaMA 一类生成模型？",
        [],
    )
    sep_answer = build_extractive_answer("为什么 [SEP] 边界标记能够提升混合文本检测效果？", [])
    ece_answer = build_extractive_answer("Temperature Scaling 与 ECE 在本文中分别说明什么？", [])

    assert "BERT" in bert_answer and "判别任务" in bert_answer
    assert "[SEP]" in sep_answer and "93.84%" in sep_answer
    assert "ECE" in ece_answer and "0.0034" in ece_answer


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


def test_project_qa_rejects_general_out_of_scope_questions(tmp_path):
    (tmp_path / "docs").mkdir(parents=True)
    (tmp_path / "docs" / "PROJECT.md").write_text(
        "本项目是中文 AI 文本检测系统，使用 BERT 模型。", encoding="utf-8"
    )

    index = ProjectKnowledgeIndex(root=tmp_path, patterns=("docs/*.md",))
    index.refresh()

    weather_hits = index.search("今天天气怎么样？", top_k=3)
    code_hits = index.search("你能帮我分析一下我写的代码有没有 bug 吗？", top_k=3)
    answer = build_extractive_answer("今天天气怎么样？", weather_hits)

    assert weather_hits == []
    assert code_hits == []
    assert "不属于当前毕业设计项目知识库范围" in answer


def test_project_qa_paraphrase_queries_route_to_kb_sources():
    index = ProjectKnowledgeIndex(root=PROJECT_ROOT)
    index.refresh()

    data_hits = index.search("你的训练数据有多少条？哪些来源？", top_k=2)
    pretrained_hits = index.search("为什么不用更新的预训练模型？", top_k=2)

    allowed_paths = {
        "docs/project/DEFENSE_KB_CURATED.md",
        "docs/project/ADVISOR_ACADEMIC_QA.md",
    }
    assert data_hits
    assert pretrained_hits
    assert data_hits[0].chunk.path in allowed_paths
    assert "AI_TEXT_TESTING_GUIDE" not in data_hits[0].chunk.path
    assert pretrained_hits[0].chunk.path in allowed_paths
    assert "英文文本" not in pretrained_hits[0].chunk.section


def test_project_qa_contextual_followup_retrieves_baseline_comparison():
    index = ProjectKnowledgeIndex(root=PROJECT_ROOT)
    index.refresh()

    contextual_question = build_contextual_retrieval_query(
        "那它和别的方法比呢？",
        [{"role": "user", "content": "刚才说的是 BERT 主模型。"}],
    )
    hits = index.search(contextual_question, top_k=2)
    answer = build_extractive_answer(contextual_question, hits)

    assert hits
    assert hits[0].chunk.path in {
        "docs/project/DEFENSE_KB_CURATED.md",
        "docs/project/ADVISOR_ACADEMIC_QA.md",
    }
    assert "BERT-BiGRU" in answer
    assert "99.28%" in answer


def test_build_extractive_answer_handles_model_size_before_metric_template():
    answer = build_extractive_answer("你这模型多大？要多少显存才能跑？", [])

    assert "391 MB" in answer
    assert "672 MB" in answer
    assert "三集平均准确率" not in answer


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


def test_project_qa_markdown_heading_context_improves_retrieval(tmp_path):
    (tmp_path / "docs" / "project").mkdir(parents=True)

    (tmp_path / "docs" / "project" / "ACADEMIC.md").write_text(
        "# 总览\n\n项目简介。\n\n"
        "## Temperature Scaling 与 ECE\n\n"
        "Temperature Scaling 用于后置概率校准，ECE 用于衡量置信度偏差。\n",
        encoding="utf-8",
    )

    index = ProjectKnowledgeIndex(root=tmp_path, patterns=("docs/project/*.md",))
    index.refresh()
    hits = index.search("Temperature Scaling 与 ECE 在这里说明什么", top_k=1)

    assert hits
    assert hits[0].chunk.path == "docs/project/ACADEMIC.md"
    assert hits[0].chunk.section == "Temperature Scaling 与 ECE"


def test_project_qa_excludes_historical_noise_sources(tmp_path):
    (tmp_path / "docs" / "project").mkdir(parents=True)
    (tmp_path / "docs" / "archive").mkdir(parents=True)
    (tmp_path / "docs" / "thesis").mkdir(parents=True)

    (tmp_path / "docs" / "project" / "DEFENSE_KB_CURATED.md").write_text(
        "当前权威答辩知识库。", encoding="utf-8"
    )
    (tmp_path / "docs" / "project" / "FINAL_RESULTS.md").write_text(
        "过时结果。", encoding="utf-8"
    )
    (tmp_path / "docs" / "archive" / "OLD.md").write_text("归档文档。", encoding="utf-8")
    (tmp_path / "docs" / "thesis" / "chapter_template.md").write_text(
        "模板文档。", encoding="utf-8"
    )

    sources = discover_project_sources(root=tmp_path, patterns=("docs/**/*.md",))
    source_paths = {path.relative_to(tmp_path).as_posix() for path in sources}

    assert "docs/project/DEFENSE_KB_CURATED.md" in source_paths
    assert "docs/project/FINAL_RESULTS.md" not in source_paths
    assert "docs/archive/OLD.md" not in source_paths
    assert "docs/thesis/chapter_template.md" not in source_paths


def test_live_repo_defense_questions_hit_curated_sources():
    index = ProjectKnowledgeIndex(root=PROJECT_ROOT)
    index.refresh()

    questions = [
        "本项目当前推荐模型是什么？核心指标有哪些？",
        "为什么本文选择 BERT 而不是 GPT 作为主模型？",
        "V11c 相比 V10 的提升主要来自哪些因素？",
        "数据治理到底做了什么？",
        "混合文本检测与边界定位能力体现在哪里？",
        "本项目目前的主要局限性是什么？",
    ]
    allowed_paths = {
        "docs/project/DEFENSE_KB_CURATED.md",
        "docs/project/ADVISOR_ACADEMIC_QA.md",
        "docs/project/DEFENSE_CURRENT_STATUS.md",
    }

    for question in questions:
        hits = index.search(question, top_k=1)
        assert hits, question
        assert hits[0].chunk.path in allowed_paths, (question, hits[0].chunk.path)


def test_build_code_symbol_index_extracts_python_and_ts_symbols(tmp_path):
    (tmp_path / "api").mkdir(parents=True)
    (tmp_path / "frontend" / "app").mkdir(parents=True)

    (tmp_path / "api" / "api.py").write_text(
        "def build_project_qa_messages(question, evidence_blocks):\n    return []\n\nclass HybridTextDetector:\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "frontend" / "app" / "demo.tsx").write_text(
        "export function DemoPanel() { return null }\nconst FancyThing = memo(function FancyThing() { return null })\n",
        encoding="utf-8",
    )

    symbol_index = build_code_symbol_index(tmp_path)

    assert "build_project_qa_messages" in symbol_index
    assert symbol_index["build_project_qa_messages"][0].path == "api/api.py"
    assert "hybridtextdetector" in symbol_index
    assert "demopanel" in symbol_index
