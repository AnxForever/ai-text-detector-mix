# V11c 训练集与能力分析（2026-02-13）

## 1) 训练集结构（`train_v11c_candidate.csv`）

- 样本总量：63,187
- 标签分布：Human 30,381（48.08%）/ AI 32,806（51.92%）
- 来源数：137（含 `unknown` 1,637 条）
- 平均长度：686 字符（中位数 313）
- 长度分位：P90=1888，P95=2345，P99=3205

### 1.1 按标签的长度特征

- Human：均值 400，中位数 201，P90=1014
- AI：均值 951，中位数 595，P90=2269
- 结论：AI 明显更长（约 2.38x）

### 1.2 长度桶分布（训练集）

- `0-64`：3,294（AI占比 9.35%）
- `64-128`：9,034（AI占比 45.64%）
- `128-256`：16,059（AI占比 33.20%）
- `256-512`：10,270（AI占比 56.57%）
- `512-1024`：8,824（AI占比 51.14%）
- `1024-2048`：10,680（AI占比 76.30%）
- `2048+`：5,026（AI占比 90.95%）

### 1.3 文本类型（按 source 推断）

- instruction_ai_generation：15,329（100% AI）
- qa_dialogue_human：14,887（100% Human）
- qa_dialogue_ai：12,790（100% AI）
- news_report：8,580（100% Human）
- long_text_ai_supplement：2,131（100% AI）
- unknown：1,637（AI占比 65.49%）

注：`text_type/domain_tag` 为 source 规则推断，不是原始标注列。

## 2) V11c 当前能力（公平三集）

- 三集平均：98.56%
- core_v1_test_clean：97.98%
- independent_data：98.57%
- merged_v2_val_clean：99.13%

### 2.1 已知强项（independent_data 按 source）

- Toutiao_News / Toutiao_news_tech / finance / edu：100%
- real_ai_gpt-4 / gpt-5 / glm-4.7 / llama-3.1-405b：100%
- real_ai_gemini-3-pro-preview：100%

### 2.2 已知短板

- real_ai_gemini-3-pro-preview-search：87.5%（7/8）
- formal_collected：96.5%
- external_m4_qazh：95.92%

### 2.3 长度切片（公平三集合并）

- `0-64`：100.00%（589）
- `64-128`：97.27%（440）← 最弱桶
- `128-256`：98.26%（403）
- `256-512`：98.07%（362）
- `512+`：99.01%（805）

## 3) 基于“强项”构建的新测试集（已执行）

构建文件：

- `datasets/eval/custom/v11c_strength_probe_2026-02-13.csv`
- `datasets/eval/custom/v11c_strength_probe_2026-02-13_summary.json`

构建策略：

- 从 `datasets/external/extracted_for_training.json` 抽样
- 目标类型：`news_article`、`qa_answer`、`qa`
- 与 `train_v11c_candidate.csv` 做 SHA1 精确去重（防止训练泄漏）
- 最终 1,199 条，近乎 1:1 标签平衡

评测结果：

- 总体 Acc：84.65%（FN=49, FP=135）
- `qa_answer`：98.83%（600）
- `qa`：84.62%（299）
- `news_article`：56.33%（300，几乎全部是 FP）

结论：

- “擅长新闻”在新来源（LCSTS 新闻）上并未泛化，存在明显分布迁移风险。
- V11c 对 QA 风格更稳，对某些新闻摘要风格不稳。

## 4) 对你当前目标（单文本检测）的建议

- 生产仍用 V11c 没问题（总体最优）。
- 但答辩时必须说明：跨来源新闻体裁存在波动，不宜宣称“新闻场景全面稳定”。
- 下一步优先补 `LCSTS`/摘要体 Human，做专门 FP 压制回归测试。
