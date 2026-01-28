# 数据集问题分析与重建建议

本报告基于当前仓库中的数据集说明与构建脚本，整理导致模型效果异常的主要风险点，并给出可直接落地的重建方向。

## 1. 关键问题定位（基于现有构建流程）

### 1.1 混合文本（带 `[SEP]`）被直接混入二分类主训练
当前 `combined_v2` 的构建把混合文本（C2/C3/C4 + `[SEP]`）与主数据合并后再整体切分。这样会把「显式边界符号」当成捷径，模型容易学到“看到 `[SEP]` 就是 AI”，导致在真实场景中迁移失效。脚本中直接读取 `hybrid_dataset_with_sep.csv` 并合并到 `combined_v2`，没有隔离训练/测试用途，也没有在主训练中剔除 `[SEP]`。此外，`test_hybrid_only.csv` 是从同一份混合集合切出来的，仍然可能携带同源分布偏差。.【F:scripts/data_cleaning/rebuild_combined_v2.py†L10-L49】

### 1.2 数据集重新切分破坏了原始 train/val/test 独立性
`final_clean` 的 train/val/test 被拼接成一个大表后重新分割，这会让原本隔离的数据重新混合，产生潜在泄漏（例如同源近重复进入不同分割）。同样的流程在 `merge_batch_data.py` 中也重复出现。只做按 `label` 的分层采样，而非按来源/模板/风格/长度/模型家族分层，会让训练与评测之间出现“同源同风格同模板”的隐性泄漏。.【F:scripts/data_cleaning/rebuild_combined_v2.py†L6-L33】【F:scripts/data_cleaning/merge_batch_data.py†L37-L66】

### 1.3 训练集缺乏风格/领域/长度元数据与配额控制
`combined_v2` 的 CSV 只有 `text,label` 字段，无法追踪来源模型、风格、领域、prompt 模板、长度桶等关键属性，也就无法进行“按风格/长度/来源配额的可控采样”。这会导致训练集中某些样式占比过高（如短文本、解释式写法），模型学习到“格式偏好”而不是“AI 痕迹”。这在当前仓库的数据格式描述里已经固化。.【F:DATA_AND_MODELS.md†L206-L216】

### 1.4 混合数据集与二分类指标被混在一个“高分”结果里
文档中提到 `bert_v2_with_sep` 的整体准确率接近 99%，并强调对 C2 检测提升，但该模型在训练时已经引入 `[SEP]`，指标存在「边界符号带来的捷径」风险。若线上文本不带 `[SEP]`，这些指标很可能无法复现。.【F:DATA_AND_MODELS.md†L39-L58】

### 1.5 缺乏 OOD 评估（模型留出、风格留出）
当前评估脚本默认使用 `combined_v2` 的 test 集，并在同分布数据上评估。没有显式的模型留出（如训练集中完全不含 Gemini，再用 Gemini 评测）或风格留出（如技术文档式只在测试集出现）机制。这样难以验证“外推能力”。.【F:scripts/evaluation/eval_complete.py†L58-L86】

## 2. 直接可执行的重建方向（与项目现状对齐）

### 2.1 建议将二分类训练与混合检测拆成“三轨制”
- **主二分类训练集**：剔除所有含 `[SEP]` 或混写样本，仅保留纯 Human/AI。
- **Mixed-Test**：保留 C2/C3/C4 作为鲁棒性测试集合，不参与主训练。
- **Mixed-Train（可选）**：若要支持混写检测，再单独构建三分类或多任务模型。

这能避免模型把 `[SEP]` 当捷径，同时也保留混写的评测价值。当前构建脚本可作为参考，但需要把混合数据从主训练集剔除并单独保存。.【F:scripts/data_cleaning/rebuild_combined_v2.py†L10-L52】

### 2.2 重新切分策略：先去重再切分，切分按来源/风格/长度分层
建议把切分逻辑从“按 label 分层”升级为“按来源模型 + 风格 + 长度桶”分层，并在切分前做精确与近重复去重。这样能最大限度降低同源泄漏，避免高估指标。当前流程只按 label 分层，建议替换或新增分层字段。.【F:scripts/data_cleaning/rebuild_combined_v2.py†L27-L33】

### 2.3 增加元数据字段，保证可控采样
建议每条样本增加以下字段：
- `style`（技术文档/列表要点/解释叙述/学术/对话/README）
- `domain`（产品/教育/医疗/科研/工程等）
- `length_bucket`（80–200/200–500/500–1000/1000–2000/2000+）
- `source_type`（human/ai）
- `model_name` / `model_family`（AI 来源）
- `prompt_id` / `decoding`（AI 生成配置）

否则无法在训练与评测中做“风格与长度配额”，也无法定位劣质模板。当前 `combined_v2` 的 CSV 格式仅支持 `text,label`，需要扩展。.【F:DATA_AND_MODELS.md†L206-L216】

### 2.4 拆分评估集：同分布 + 风格外推 + 模型外推
建议把 `combined_v2/test.csv` 拆为三类评测集：
- **ID（同分布）**：和训练配额一致
- **Style-OOD**：留出某类风格（如技术文档）只在测试集出现
- **Model-OOD**：留出某个模型家族（如 Gemini/GPT）只在测试集出现

当前评估脚本默认使用同分布 test，建议补充两个独立评测入口。.【F:scripts/evaluation/eval_complete.py†L58-L86】

## 3. 可以直接落地的修改建议（下一步优先级）

1. **移除主训练中所有带 `[SEP]` 的样本**，并将混写样本只保留在 `Mixed-Test` 中。
2. **扩展数据字段**（至少包括 `style`、`length_bucket`、`model_name`），为可控采样做准备。
3. **重新切分数据集**：先全局去重，再按「来源×风格×长度」分层切分。
4. **新增两套评测集**：Model-OOD 与 Style-OOD，明确评估外推能力。

## 4. 备注：当前仓库结构对齐点
- `combined_v2` 作为主训练集的“入口”已经固定，重建后仍可复用原训练脚本。.【F:scripts/training/train_v2_simple.py†L70-L73】
- 混写样本已经具备分类标记和分离脚本，可直接作为 `Mixed-Test` 的基础。.【F:scripts/data_cleaning/rebuild_combined_v2.py†L40-L52】

---

如需我继续把“配额表 + 风格模板库 + 分割规则”落地成具体脚本（自动标注 `style`、`length_bucket`、`model_name`），请告诉我你当前的原始数据来源文件结构，我可以直接按你已有目录生成重建脚本。
