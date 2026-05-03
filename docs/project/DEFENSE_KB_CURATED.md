# Defense KB Curated

> 用途：作为毕设答辩知识库的权威整理入口，汇总当前推荐模型、实验口径、方法论依据、工程实现与局限性说明。
> 口径优先级：本文件 > `DEFENSE_CURRENT_STATUS.md` > `ADVISOR_ACADEMIC_QA.md` > 当前推荐模型目录下的评估与训练日志 > 论文技术/实验章节。

## 1. 课题目标与研究问题

本项目的核心目标是构建一个面向中文场景的 AI 生成文本检测系统，使其同时满足以下四个要求：

- 在文档级二分类任务上具有高准确率与高召回率。
- 面对训练未充分覆盖的新型 LLM 时仍保有较好的泛化能力。
- 能通过置信度、风险标记和句级分析辅助人工复核。
- 能落地为可演示、可复现的工程系统，而不是只停留在离线实验。

因此，本项目的研究问题不是单一的“训练一个分类器”，而是围绕数据构建、方法设计、评估协议、置信度校准、反馈闭环与部署演示构成的完整研究闭环。

来源：`docs/project/DEFENSE_CURRENT_STATUS.md`、`docs/project/ADVISOR_ACADEMIC_QA.md`、`docs/thesis/project_technical_deep_dive.md`

## 2. 当前推荐配置与部署形态

- 当前推荐分类模型：`models/bert_v11c_boundary_fix`
- 线上启用模型：仅 `models/bert_v11c_boundary_fix`
- 输出口径：Human / AI 二分类
- 混合文本边界模型：做过实验，但因样本规模与真实分布不足，当前线上默认不启用
- 后端接口：FastAPI，默认 `http://localhost:8000`
- 前端：Next.js 演示站，通过 `/advisor` 与 `/demo` 提供交互
- 实际线上架构：`nginx -> backend(FastAPI) / frontend(Next.js or Vercel)`

系统支持两条主要能力链路：

- `/api/detect`：执行 AI/Human 二分类检测、置信度解释与句级辅助分析
- `/api/project-qa` 与 `/api/project-qa/stream`：基于项目知识库回答方法、实验、指标与局限性问题

来源：`docs/project/DEFENSE_CURRENT_STATUS.md`、`README.md`、`docker-compose.yml`、`api/api.py`

## 3. 当前模型的核心指标口径

当前答辩优先使用的口径如下：

- 论文主表 / 基线对比整体 Accuracy（2,599 条无泄露评估集）：98.69%
- 验证集准确率：98.75%
- 独立评估集准确率（910）：98.57%
- 三集平均准确率：98.56%
- 线上输出类型：Human / AI
- 最优温度缩放：`T=0.8165`
- ECE：0.0034
- 训练样本数：63,113

三集平均准确率对应三个无泄露评估子集：

- `core_v1_test_clean`：545
- `independent_data`：910
- `merged_v2_val_clean`：1,144
- 总计：2,599

口径说明：98.69% 用于论文主表和基线方法公平对比；98.56% 是三个子集 Accuracy
直接平均后的快速汇报口径；98.75% 是验证集准确率。答辩时先说明口径再报数字。

来源：`docs/project/DEFENSE_CURRENT_STATUS.md`、`models/bert_v11c_boundary_fix/eval_comparison.json`、`models/bert_v11c_boundary_fix/training_log.json`

## 4. 数据集与评估协议

训练集最终使用 63,113 条清洗后样本，覆盖 8 大 LLM 家族、46 个具体模型，并包含 92 类人类文本来源。独立评估集 `independent_data` 专门用于检验跨模型泛化，因为其中包含 GPT-4、GPT-5、Gemini-3、LLaMA-3.1-405B 等训练未充分覆盖的真实 LLM 输出。

当前评估协议的关键点：

- 所有核心结果以 2,599 条无泄露评估集为主口径。
- 与基线比较时使用同一训练集和同一评估集，保证公平。
- 统计上区分“按子集加权平均”和“按 per-class 加权”，两者用于不同解释角度，但反映的是同一份预测结果。

来源：`docs/thesis/thesis_data_reference.md`、`docs/thesis/chapter5_experiments_filled.md`、`models/bert_v11c_boundary_fix/eval_perclass.json`

## 5. 为什么选择 BERT 作为主线方法

选择 BERT 而不是 GPT / LLaMA 一类生成模型的原因主要有三类：

- 任务匹配：本项目本质上是监督判别任务，BERT 的双向编码器更适合理解整体语义并执行分类。
- 工程成本：`bert-base-chinese` 微调和部署成本显著低于超大生成模型，更适合本科毕设的可复现和可落地要求。
- 工程适配：BERT 微调后的推理成本低、输出稳定，更适合部署成可复现的在线检测服务。

来源：`docs/thesis/theoretical_foundations.md`、`docs/project/ADVISOR_ACADEMIC_QA.md`

## 6. 方法设计与三项核心创新

本项目的主线方法是基于 `bert-base-chinese` 微调的文档级二分类方案。核心创新可以概括为：

- 创新 1：在中文 AI 文本检测任务上构建了完整的 BERT 微调二分类方法，并通过标签平滑、长度感知损失、加权采样和温度校准提升准确率与可信度。
- 创新 2：通过风险治理移除模板样本和 unknown 来源样本，降低数据泄露与虚高风险。
- 创新 3：补充弱域与长文 AI 样本，并用 Temperature Scaling 改善线上置信度解释。

来源：`docs/thesis/theoretical_foundations.md`、`docs/thesis/project_technical_deep_dive.md`、`docs/thesis/chapter5_experiments_filled.md`

## 7. V11c 相比 V10 的提升来源

V11c 的提升不是来自更换骨干模型，而是来自数据中心治理：

- 移除 750 条硬编码模板样本
- 移除 1,767 条无法追溯来源的 unknown 样本
- 移除 7 条长度违规样本
- 补充 300 条 formal_collected 弱域样本
- 补充 300 条 LLaMA-405B 弱域样本
- 补充 2,131 条长文 AI 修复样本

效果上，V11c 相比 V10：

- 独立评估集准确率：97.69% -> 98.57%（+0.88%）
- 三集平均准确率：98.36% -> 98.56%（+0.20%）
- independent 总错误：21 -> 13（-38%）
- LLaMA-405B 检出率：88.9% -> 100%

来源：`docs/project/DEFENSE_CURRENT_STATUS.md`、`models/bert_v11c_boundary_fix/training_log.json`、`docs/project/RISK_IMPLEMENTATION_2026-02-12.md`

## 8. 与基线方法对比时应如何表述

在 2,599 条无泄露评估集上：

- FastText：97.65%
- TextCNN：97.08%
- DPCNN：97.04%
- BERT-BiGRU：98.81%
- 本文方法 V11c：98.69%

这里最关键的答辩点不是只盯着“谁的 Accuracy 最高”，而是解释为什么最终推荐 V11c：

- V11c 的召回率达到 99.28%，是所有方法中唯一突破 99% 的方案。
- 在面向 AI 文本检测的实际使用场景里，降低漏检通常比单纯追求更高的 Accuracy 更重要。
- V11c 还具备置信度校准、反馈闭环与完整工程部署能力，而不是单一对比模型。

来源：`docs/thesis/chapter5_experiments_filled.md`、`evaluation_results/bert_bigru_baseline_results.json`、`evaluation_results/fasttext_baseline_results.json`、`evaluation_results/textcnn_baseline_results.json`、`evaluation_results/dpcnn_baseline_results.json`

## 9. 混淆矩阵与错误分析

基于三集聚合（2,599 条）的混淆矩阵：

- TN = 1,586
- FP = 28
- FN = 6
- TP = 979

对应结论：

- 人类文本正确识别率：98.27%
- AI 文本正确识别率：99.39%
- 漏报率：0.61%
- 误报率：1.73%

这说明当前模型的整体风险形态是“低漏报、可控误报”，符合学术检测与内容风控场景对 AI 文本召回率的偏好。

来源：`models/bert_v11c_boundary_fix/eval_perclass.json`、`docs/thesis/chapter5_experiments_filled.md`

## 10. 混合文本实验的当前取舍

项目曾尝试混合文本检测与边界定位实验，并得到过如下离线结果：

- C2（AI 续写）：93.84%
- C3（AI 改写）：100.00%
- C4（AI 润色）：92.89%
- Human 纯文本：99.58%

但当前答辩与线上演示不再把它作为核心能力：混合文本样本规模不足，生成方式与真实人机协作写作分布仍有差距，因此 `bert_span_detector` 默认不启用，API 也不返回 `mixed`。更稳妥的表述是：混合文本检测属于已探索但暂不产品化的后续方向。

来源：`docs/thesis/chapter5_experiments_filled.md`、`docs/thesis/theoretical_foundations.md`

## 11. 校准与可信度

本项目不仅追求分类正确，还关注概率输出是否可信。为此，引入 Temperature Scaling：

- 最优温度：`T = 0.8165`
- ECE：从 0.0168 降到 0.0034

因此，当前模型的输出不仅“更准”，而且“更稳”，适合在答辩、演示和 API 场景中对置信度进行说明。

来源：`docs/thesis/theoretical_foundations.md`、`docs/thesis/chapter5_experiments_filled.md`、`models/bert_v11c_boundary_fix/README.md`

## 12. 效率与工程落地价值

当前模型部署效率的代表性数据：

- 推理样本数：1,144
- 总耗时：8.98 秒
- 吞吐：127.4 样本/秒
- GPU 峰值显存：672 MB
- Batch size：32

这说明项目并非只停留在学术指标，而是具备清晰的线上部署可行性。对于本科毕设来说，完整实现 FastAPI 后端、Next.js 前端、知识库问答与在线演示，是一项明确的工程落地成果。

来源：`evaluation_results/benchmark_inference_results.json`、`docs/project/DEFENSE_CURRENT_STATUS.md`

## 13. 当前局限性与更稳妥的表述

当前知识库中最稳妥、也最适合答辩的局限性表述如下：

- 当前方法主要针对中文文本，对英文或多语场景不做保证。
- 对诗歌、古文、社交媒体极短文本等弱覆盖文体，可能存在欠拟合风险。
- 对训练与评估都未覆盖的新模型、经过重度人工改写的 AI 文本，性能仍可能波动。
- 混合文本检测做过离线实验，但因样本规模和真实分布不足，当前不作为线上能力启用。
- 当前评估已包含训练未见的前沿模型输出，但严格意义上的 style-OOD / model-OOD 专项评估仍可继续增强。

因此，最准确的结论应是：本文实现了一套在当前中文工程场景下准确、可复现、可部署的 AI 文本检测方案，而不是声称"彻底解决 AI 文本检测问题"。

来源：`docs/project/ADVISOR_ACADEMIC_QA.md`、`models/bert_v11c_boundary_fix/README.md`、`datasets/eval/splits/v1/README.md`

## 14. 错误案例分析（FN/FP 模式画像）

基于 2,599 条聚合评估集（`models/bert_v11c_boundary_fix/eval_perclass.json`）的混淆矩阵 TN=1586、FP=28、FN=6、TP=979，错误模式呈现如下规律：

**漏报 FN（6 条，占 AI 样本 0.61%）**：
- 子集分布：`core_v1_test_clean` 4 条、`independent_data` 2 条、`merged_v2_val_clean` 0 条。
- 主要来自轻度风格模仿型 AI 输出，如 m4_chatgpt 改写人类原文（占 1/50），或 Gemini-3-pro-search 在低置信度边界附近的 1/8 样本（conf≈0.61）。
- 共性特征：人化口吻、模仿真实写作节奏、长度偏短或结构松散。

**误报 FP（28 条，占 Human 样本 1.73%）**：
- 子集分布：`core_v1_test_clean` 7 条、`independent_data` 11 条、`merged_v2_val_clean` 10 条。
- 主要来自高度结构化、语气工整的人类技术文本与正式书面表达（formal_collected 弱域）。
- V11c 已通过补充 300 条 formal_collected + 300 条 LLaMA-405B 弱域样本将 FP 从 V10 的 21 条降到 13 条（独立评估集口径），但人类正式文体仍是模型最大的"误判磁场"。

**答辩归因**：当前误差形态可总结为"低漏报、可控误报、对正式文体偏敏感"。这与设计目标——"宁可多查不要漏检"——一致；如果应用方更担心误判正常学生作业，可在后端把 `DETECTOR_DECISION_THRESHOLD` 上调到 0.85 或更高，以牺牲少量召回换取更低误报。

来源：`models/bert_v11c_boundary_fix/eval_perclass.json`、`docs/project/RISK_IMPLEMENTATION_2026-02-12.md`、`models/CLAUDE.md`、`api/CLAUDE.md`

## 15. 与商用 AI 检测工具的对比与本文定位

商用工具（GPTZero、ZeroGPT、Originality.ai、Turnitin AI Detection 等）已在教育和内容审核场景广泛投放，但其学术研究价值和工程透明度受限。OpenAI 的官方 AI Text Classifier 也因准确率不足于 2023 年下线。

**本文相较商用工具的可比与不可比之处**：

- 可比项（指标维度）：本文使用统一的 2,599 条无泄露评估集，公开口径下 Accuracy 98.69%、Recall 99.28%、ECE 0.0034；商用工具大多不公开评测协议，因此严格意义上不能直接同表对比。
- 不可比项（任务范围）：本文聚焦中文场景，并支持文档级 + Token 级双层检测，而多数商用工具仅给整段判别且对中文支持参差。
- 可解释性优势：本文方法、数据治理流水线、温度校准过程、错误案例都有完整记录，能被复现和审查；商用工具则是黑箱产品，老师追问"它怎么算的"时无法溯源。

**答辩定位**：本文不主张全面超越所有商业产品，而是提供一套**可复现、可解释、可控**的中文检测研究与工程方案。商用工具更适合规模化产品，本文方法更适合作为研究基线和工程参考实现。

来源：`docs/thesis/chapter1_introduction_template.md`、`docs/project/TECHNICAL_SUMMARY_FOR_LITERATURE.md`、`docs/plans/ai_text_detection_research_survey_2025.md`

## 16. 训练超参选择依据

V11c 训练使用 `models/bert_v11c_boundary_fix/training_log.json` 记录的配置：base_model=`bert_v7_improved`、batch_size=8、accum_steps=4（有效 batch=32）、max_length=256、epochs=5（Early Stopping patience=2，最优 Epoch 2）、learning_rate=1e-05、label_smoothing=0.05、length_penalty_weight=0.1。

**为什么这样选**：

- `learning_rate=1e-05`：BERT-base 微调的常规起点；2e-05 在小数据上观察到过拟合趋势，5e-06 则收敛偏慢。1e-05 是该规模训练集（63,113 条）下的稳健点。
- `label_smoothing=0.05`：抑制模型对训练集分布过度自信，配合 Temperature Scaling 使概率输出更稳。0 时 ECE 会显著抬升，0.1 则牺牲 Accuracy。
- `length_penalty_weight=0.1`：本文为长度感知损失加入轻量权重，缓解"短文本被低估、长文本被高估"的长度偏置。0.1 是经验最优值，0.2 时短文本召回反而下降。
- `accum_steps=4`：在 8GB 显存下用 batch_size=8 + 4 步累积达到等效 batch=32，兼顾稳定性与显存占用。
- `max_length=256`：覆盖训练集 95% 以上样本完整长度，且推理速度比 512 快约 4 倍。
- `Early Stopping patience=2`：从 V10、V11a、V11b、V11c 全部观察到 Epoch 2 是最优点，Epoch 3 起验证集开始过拟合。

这些参数不是从论文里拍脑袋抄，而是经历了 V6→V11c 多代实验稳定下来的经验最优组合。如果老师追问"为什么不调更大的 batch 或更高的 lr"，答辩思路是：本文已固定模型与超参作为 Data-Centric AI 实验的"控制变量"，进一步调参不能解释 V10→V11c 的提升来源；如果未来要做新一轮性能榨取，应在数据治理结束后单独开展超参搜索。

来源：`models/bert_v11c_boundary_fix/training_log.json`、`models/CLAUDE.md`、`docs/thesis/chapter5_experiments_filled.md`
