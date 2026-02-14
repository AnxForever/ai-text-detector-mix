# Dataset Inventory (Step 1)

Generated from file scan. Row counts are estimated from line counts for CSV/TSV/JSONL when file size <= 200 MB.
Note: text fields often contain newlines, so line-count estimates can be higher than actual record counts.
Note: datasets directory has since been reorganized; paths in this report may be outdated.

## Summary

### Files by Group

- augmented_v2: 7 files
- combined_v2: 4 files
- combined_v2_clean: 6 files
- eval_splits: 4 files
- evaluation_results: 5 files
- final_clean: 6 files
- final_clean_phrase_clean: 4 files
- hybrid: 35 files
- metadata: 3 files
- mixed_test_candidates: 5 files
- pred_probs: 2 files
- raw: 13 files
- resources: 1 files
- routed: 12 files

### Files by Extension

- .csv: 49
- .json: 40
- .jsonl: 8
- .txt: 10

## Inventory Table

| Group | Path | Ext | Size(MB) | Rows(est) | Schema/Keys (sample) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| augmented_v2 | datasets\logs\augmented_v2\generation_log_20260126_210605.txt | .txt | 0.02 |  |  | augmented |
| augmented_v2 | datasets\logs\augmented_v2\generation_log_20260126_210902.txt | .txt | 0.01 |  |  | augmented |
| augmented_v2 | datasets\logs\augmented_v2\generation_log_20260126_211114.txt | .txt | 0.00 |  |  | augmented |
| augmented_v2 | datasets\logs\augmented_v2\generation_log_20260126_211317.txt | .txt | 0.00 |  |  | augmented |
| augmented_v2 | datasets\logs\augmented_v2\generation_log_20260126_211718.txt | .txt | 0.00 |  |  | augmented |
| augmented_v2 | datasets\logs\augmented_v2\generation_log_20260126_212535.txt | .txt | 0.00 |  |  | augmented |
| augmented_v2 | datasets\logs\augmented_v2\generation_log_20260126_213018.txt | .txt | 0.00 |  |  | augmented |
| combined_v2 | datasets\active\core_v1\test.csv | .csv | 10.34 | 75075 | text, source, label, length, category | test |
| combined_v2 | datasets\mixed\hybrid\hybrid_dataset_with_sep.csv | .csv | 1.67 | 8357 | text, source, label, length, category | test, hybrid |
| combined_v2 | datasets\active\core_v1\train.csv | .csv | 83.26 | 621612 | text, source, label, length, category | train |
| combined_v2 | datasets\active\core_v1\val.csv | .csv | 10.28 | 76878 | text, source, label, length, category | val |
| combined_v2_clean | datasets\archive\_to_delete\sources_20260127\combined_v2_clean\cleaning_log.json | .json | 0.00 |  | timestamp, input_dir, output_dir, rows_before, sep_removed, phrase_removed, duplicates_removed, rows_after, split_counts |  |
| combined_v2_clean | datasets\archive\_to_delete\sources_20260127\combined_v2_clean\mixed_sep.csv | .csv | 3.09 | 10318 | text, source, label, length, category |  |
| combined_v2_clean | datasets\archive\_to_delete\sources_20260127\combined_v2_clean\phrase_removed.csv | .csv | 0.53 | 3674 | text, source, label, length, category |  |
| combined_v2_clean | datasets\archive\_to_delete\sources_20260127\combined_v2_clean\test.csv | .csv | 9.49 | 76192 | text, source, label, length, category | test |
| combined_v2_clean | datasets\archive\_to_delete\sources_20260127\combined_v2_clean\train.csv | .csv | 75.95 | 597253 | text, source, label, length, category | train |
| combined_v2_clean | datasets\archive\_to_delete\sources_20260127\combined_v2_clean\val.csv | .csv | 9.52 | 78056 | text, source, label, length, category | val |
| eval_splits | datasets\eval\splits\v1\id_test_combined_v2_clean.csv | .csv | 9.49 | 76192 | text, source, label, length, category | val, test, eval_split |
| eval_splits | datasets\eval\splits\v1\id_test_final_clean_phrase_clean.csv | .csv | 8.65 | 68373 | text, source, label, length | val, test, eval_split |
| eval_splits | datasets\eval\splits\v1\mixed_test_balanced_by_category.csv | .csv | 8.29 | 52986 | text, label, category | val, test, mixed_test, eval_split |
| eval_splits | datasets\eval\splits\v1\mixed_test_balanced_by_category_length.csv | .csv | 6.21 | 39336 | text, label, category | val, test, mixed_test, eval_split |
| final_clean | datasets\active\core_v1\all_ai.csv | .csv | 55.51 | 647918 | text, source, label, length |  |
| final_clean | datasets\active\core_v1\all_human.csv | .csv | 32.44 | 47830 | text, source, label, length |  |
| final_clean | datasets\active\core_v1\full_dataset.csv | .csv | 87.95 | 695748 | text, source, label, length |  |
| final_clean | datasets\active\core_v1\test.csv | .csv | 8.70 | 68832 | text, source, label, length | test |
| final_clean | datasets\active\core_v1\train.csv | .csv | 70.28 | 556464 | text, source, label, length | train |
| final_clean | datasets\active\core_v1\val.csv | .csv | 8.84 | 70452 | text, source, label, length | val |
| final_clean_phrase_clean | datasets\archive\_to_delete\sources_20260127\final_clean_phrase_clean\cleaning_log.json | .json | 0.00 |  | timestamp, input_dir, output_dir, summary, phrases |  |
| final_clean_phrase_clean | datasets\archive\_to_delete\sources_20260127\final_clean_phrase_clean\test.csv | .csv | 8.65 | 68373 | text, source, label, length | test |
| final_clean_phrase_clean | datasets\archive\_to_delete\sources_20260127\final_clean_phrase_clean\train.csv | .csv | 69.91 | 553503 | text, source, label, length | train |
| final_clean_phrase_clean | datasets\archive\_to_delete\sources_20260127\final_clean_phrase_clean\val.csv | .csv | 8.80 | 70227 | text, source, label, length | val |
| hybrid | datasets\mixed\hybrid\c2_batch.json | .json | 1.80 |  | list[1000] | hybrid |
| hybrid | datasets\mixed\hybrid\c2_continuation.json | .json | 0.30 |  | list[225] | hybrid |
| hybrid | datasets\mixed\hybrid\c2_fast.json | .json | 0.31 |  | list[200] | hybrid |
| hybrid | datasets\mixed\hybrid\c2_final.json | .json | 0.14 |  | list[100] | hybrid |
| hybrid | datasets\mixed\hybrid\c2_local.json | .json | 0.15 |  | list[100] | hybrid |
| hybrid | datasets\mixed\hybrid\c2_local_v2.json | .json | 0.51 |  | list[309] | hybrid |
| hybrid | datasets\mixed\hybrid\c2_local_v3.json | .json | 0.15 |  | list[100] | hybrid |
| hybrid | datasets\mixed\hybrid\c2_span_labels.json | .json | 26.09 |  |  | hybrid |
| hybrid | datasets\mixed\hybrid\c3_batch.json | .json | 2.33 |  |  | hybrid |
| hybrid | datasets\mixed\hybrid\c3_edited.json | .json | 0.21 |  | list[94] | hybrid |
| hybrid | datasets\mixed\hybrid\c3_edited_kfc.json | .json | 0.05 |  | list[50] | hybrid |
| hybrid | datasets\mixed\hybrid\c3_final.json | .json | 0.31 |  | list[200] | hybrid |
| hybrid | datasets\mixed\hybrid\c3_local_v2.json | .json | 0.36 |  | list[200] | hybrid |
| hybrid | datasets\mixed\hybrid\c3_public.json | .json | 0.05 |  | list[50] | hybrid |
| hybrid | datasets\mixed\hybrid\c4_batch.json | .json | 1.27 |  | list[500] | hybrid |
| hybrid | datasets\mixed\hybrid\c4_fast.json | .json | 0.52 |  | list[300] | hybrid |
| hybrid | datasets\mixed\hybrid\c4_final.json | .json | 0.55 |  | list[305] | hybrid |
| hybrid | datasets\mixed\hybrid\c4_local.json | .json | 0.20 |  | list[100] | hybrid |
| hybrid | datasets\mixed\hybrid\c4_local_v2.json | .json | 1.08 |  | list[400] | hybrid |
| hybrid | datasets\mixed\hybrid\c4_local_v3.json | .json | 1.17 |  | list[444] | hybrid |
| hybrid | datasets\mixed\hybrid\c4_polished.json | .json | 0.27 |  | list[122] | hybrid |
| hybrid | datasets\mixed\hybrid\c4_polished_kfc.json | .json | 0.15 |  | list[100] | hybrid |
| hybrid | datasets\mixed\hybrid\c4_polished_x666.json | .json | 0.07 |  | list[64] | hybrid |
| hybrid | datasets\mixed\hybrid\c4_public.json | .json | 0.15 |  | list[100] | hybrid |
| hybrid | datasets\mixed\hybrid\hybrid_dataset.csv | .csv | 7.13 | 49144 | text, label, category | hybrid |
| hybrid | datasets\mixed\hybrid\hybrid_dataset_expanded.csv | .csv | 12.28 | 72489 | text, label, category | hybrid |
| hybrid | datasets\mixed\hybrid\hybrid_dataset_with_sep.csv | .csv | 12.29 | 72489 | text, label, category | hybrid |
| hybrid | datasets\mixed\hybrid\merged_all.json | .json | 8.65 |  |  | hybrid |
| hybrid | datasets\mixed\hybrid\multimodel\checkpoint_deepseek-v3.json | .json | 0.22 |  | list[110] | hybrid |
| hybrid | datasets\mixed\hybrid\multimodel\checkpoint_glm-4.7.json | .json | 0.31 |  | list[160] | hybrid |
| hybrid | datasets\mixed\hybrid\multimodel\checkpoint_gpt-3.5-turbo.json | .json | 0.55 |  | list[270] | hybrid |
| hybrid | datasets\mixed\hybrid\multimodel\checkpoint_qwen3-32b.json | .json | 0.31 |  | list[160] | hybrid |
| hybrid | datasets\mixed\hybrid\test.csv | .csv | 0.71 | 4798 | text, label, category | test, hybrid |
| hybrid | datasets\mixed\hybrid\train.csv | .csv | 5.70 | 39380 | text, label, category | train, hybrid |
| hybrid | datasets\mixed\hybrid\val.csv | .csv | 0.72 | 4966 | text, label, category | val, hybrid |
| metadata | datasets\analysis\metadata\audit_logs\2026-01-27_audit_summary.json | .json | 0.02 |  | timestamp, datasets, cross_dataset_overlap | metadata |
| metadata | datasets\analysis\metadata\audit_logs\2026-01-27_cross_overlap_summary.json | .json | 0.00 |  | timestamp, counts, overlaps | metadata |
| metadata | datasets\analysis\metadata\audit_logs\2026-01-27_mixed_test_stats.json | .json | 0.00 |  | rows, label_counts, category_counts, sep_rows, phrase_rows, length_buckets | test, mixed_test, metadata |
| mixed_test_candidates | datasets\mixed\candidates\hybrid_expanded_clean.csv | .csv | 3.08 | 10345 | text, label, category | test, mixed_test, hybrid |
| mixed_test_candidates | datasets\mixed\candidates\hybrid_expanded_clean_no_phrases.csv | .csv | 3.07 | 10318 | text, label, category | test, mixed_test, hybrid |
| mixed_test_candidates | datasets\mixed\candidates\hybrid_expanded_clean_vs_final_clean.csv | .csv | 10.51 | 69716 | text, label, category | test, mixed_test, hybrid |
| mixed_test_candidates | datasets\mixed\candidates\mixed_test_balanced_by_category.csv | .csv | 8.29 | 52986 | text, label, category | test, mixed_test |
| mixed_test_candidates | datasets\mixed\candidates\mixed_test_balanced_by_category_length.csv | .csv | 6.21 | 39336 | text, label, category | test, mixed_test |
| pred_probs | datasets\analysis\pred_probs\combined_v2_clean\test.csv | .csv | 9.61 | 76192 | text, source, label, length, category, pred_ai_prob | test, pred_probs |
| pred_probs | datasets\analysis\pred_probs\combined_v2_clean\val.csv | .csv | 9.64 | 78056 | text, source, label, length, category, pred_ai_prob | val, pred_probs |
| raw | datasets\raw\COLLECTION_GUIDE.json | .json | 0.00 |  | created_at, datasets, instructions, notes | raw |
| raw | datasets\raw\HC3-Chinese\all.jsonl | .jsonl | 20.72 | 12853 | question, human_answers, chatgpt_answers, source | raw |
| raw | datasets\raw\HC3-Chinese\baike.jsonl | .jsonl | 4.77 | 4617 | question, human_answers, chatgpt_answers | raw |
| raw | datasets\raw\HC3-Chinese\finance.jsonl | .jsonl | 1.55 | 689 | question, human_answers, chatgpt_answers | raw |
| raw | datasets\raw\HC3-Chinese\law.jsonl | .jsonl | 0.50 | 372 | question, human_answers, chatgpt_answers | raw |
| raw | datasets\raw\HC3-Chinese\medicine.jsonl | .jsonl | 1.53 | 1074 | question, human_answers, chatgpt_answers | raw |
| raw | datasets\raw\HC3-Chinese\nlpcc_dbqa.jsonl | .jsonl | 1.93 | 1709 | question, human_answers, chatgpt_answers | raw |
| raw | datasets\raw\HC3-Chinese\open_qa.jsonl | .jsonl | 6.23 | 3293 | question, human_answers, chatgpt_answers | raw |
| raw | datasets\raw\HC3-Chinese\psychology.jsonl | .jsonl | 3.97 | 1099 | question, human_answers, chatgpt_answers | raw |
| raw | datasets\raw\ai\parallel_dataset_cleaned.csv | .csv | 53.22 | 620686 | text_id, text_content, source_api, source_model, attribute, topic, genre, role, style, constraint, prompt, combination_quality, generation_quality, length, timestamp | raw |
| raw | datasets\raw\ai\parallel_dataset_cleaning_report.json | .json | 0.00 |  | input_file, output_file, total_rows, kept_rows, removed_rows, removed_pct, head_chars, reason_counts, samples, generated_at | raw |
| raw | datasets\raw\human_texts\sample_1000_texts.csv | .csv | 0.81 | 1000 | text, source, category, length, timestamp | raw |
| raw | datasets\raw\human_texts\thucnews_real_human_9000.csv | .csv | 23.32 | 9000 | text, source, attribute, topic, length, timestamp | raw |
| routed | datasets\analysis\routed\combined_v2_clean\pools\core.csv | .csv | 90.31 | 686866 | text, source, label, length, category, _split, style, y_main, length_bucket, Q, D, y_conf, q_flags, d_flags, y_evidence, routing, rule_version | routed |
| routed | datasets\analysis\routed\combined_v2_clean\pools\hard.csv | .csv | 0.00 | 0 | text, source, label, length, category, _split, style, y_main, length_bucket, Q, D, y_conf, q_flags, d_flags, y_evidence, routing, rule_version | routed |
| routed | datasets\analysis\routed\combined_v2_clean\pools\reject.csv | .csv | 0.00 | 3 | text, source, label, length, category, _split, style, y_main, length_bucket, Q, D, y_conf, q_flags, d_flags, y_evidence, routing, rule_version | routed |
| routed | datasets\analysis\routed\combined_v2_clean\pools\review.csv | .csv | 8.69 | 64632 | text, source, label, length, category, _split, style, y_main, length_bucket, Q, D, y_conf, q_flags, d_flags, y_evidence, routing, rule_version | routed |
| routed | datasets\analysis\routed\combined_v2_clean\scored_all.csv | .csv | 99.00 | 751501 | text, source, label, length, category, _split, style, y_main, length_bucket, Q, D, y_conf, q_flags, d_flags, y_evidence, routing, rule_version | routed |
| routed | datasets\analysis\routed\combined_v2_clean\summary.json | .json | 0.00 |  | timestamp, dataset_name, rows, pool_counts, q_flag_counts, d_flag_counts, y_conflicts, rule_version | routed |
| routed | datasets\analysis\routed\final_clean_phrase_clean\pools\core.csv | .csv | 90.08 | 686866 | text, source, label, length, _split, category, style, y_main, length_bucket, Q, D, y_conf, q_flags, d_flags, y_evidence, routing, rule_version | routed |
| routed | datasets\analysis\routed\final_clean_phrase_clean\pools\hard.csv | .csv | 0.00 | 0 | text, source, label, length, _split, category, style, y_main, length_bucket, Q, D, y_conf, q_flags, d_flags, y_evidence, routing, rule_version | routed |
| routed | datasets\analysis\routed\final_clean_phrase_clean\pools\reject.csv | .csv | 0.00 | 3 | text, source, label, length, _split, category, style, y_main, length_bucket, Q, D, y_conf, q_flags, d_flags, y_evidence, routing, rule_version | routed |
| routed | datasets\analysis\routed\final_clean_phrase_clean\pools\review.csv | .csv | 0.99 | 5234 | text, source, label, length, _split, category, style, y_main, length_bucket, Q, D, y_conf, q_flags, d_flags, y_evidence, routing, rule_version | routed |
| routed | datasets\analysis\routed\final_clean_phrase_clean\scored_all.csv | .csv | 91.07 | 692103 | text, source, label, length, _split, category, style, y_main, length_bucket, Q, D, y_conf, q_flags, d_flags, y_evidence, routing, rule_version | routed |
| routed | datasets\analysis\routed\final_clean_phrase_clean\summary.json | .json | 0.00 |  | timestamp, dataset_name, rows, pool_counts, q_flag_counts, d_flag_counts, y_conflicts, rule_version | routed |
| evaluation_results | evaluation_results\evaluation_report.txt | .txt | 0.00 |  |  | val |
| evaluation_results | evaluation_results\final_report.txt | .txt | 0.00 |  |  | val |
| evaluation_results | evaluation_results\length_aware_evaluation.csv | .csv | 0.00 | 5 | bin, bin_range_str, accuracy, ai_accuracy, human_accuracy, precision, recall, f1, confusion_matrix, sample_count, ai_count, human_count | val |
| evaluation_results | evaluation_results\length_aware_evaluation.json | .json | 0.00 |  | evaluation_date, model_path, test_csv, length_bins, bin_results, statistics | val |
| evaluation_results | evaluation_results\model_comparison.json | .json | 0.00 |  | BERT V2 | val |
| resources | resources\cilin.txt | .txt | 0.00 |  |  |  |
