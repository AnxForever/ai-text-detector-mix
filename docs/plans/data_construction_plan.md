# 数据构建计划

> 更新时间: 2026-01-27
> 目标: 构建 60K 中文二分类主训练集（Human:AI=1:1）+ OOD 测试集 + Mixed-Test（Security-Test 可选）+ 高风险域 OOD_HS

---

## 一、范围与目标

- 主任务: Human vs AI 文档级二分类
- 混合文本: 单独 Mixed-Test，不进入主训练集
- 安全场景: 钓鱼/诈骗文本可选 Security-Test（不进入主训练集）
- 高风险域: 法律/金融独立 OOD_HS（不进主训练/不调参）
- 以“场景”为主轴，以“文风”为副轴进行配额

成功标准:
- 主任务准确率稳定 95%+，结构化文风（列表/说明书）明显提升
- Style-OOD / Model-OOD 可解释评估（Security-Test 可选扩展）
- 数据集可追溯、可复现、可审计

---

## 二、输入与现状

### 现有数据集
- datasets/active/core_v1（合并基线）
- datasets/archive/combined_v2（噪声与 [SEP] 混入）
- datasets/mixed/hybrid（混合文本样本）

### 当前问题摘要
- 结构化文本识别显著偏低
- combined_v2 含 [SEP] 3.1%
- 长度分布偏短（<200 占 38%）

---

## 三、输出结构（拟定）

- datasets/active/core_v2/（主训练集）
  - train.jsonl / val.jsonl / test.jsonl
  - full_dataset.csv
- datasets/eval/splits/v2/（ID / Style-OOD / Model-OOD）
- datasets/eval/mixed_test/（Mixed-Test）
- datasets/eval/security_phishing/（Security-Test，可选）
- datasets/eval/ood_high_stakes/（OOD_HS 高风险域子集）
- datasets/analysis/metadata/（配额表、采样日志、版本说明）

---

## 四、最小元数据字段

```json
{
  "id": "zh_edu_000001",
  "text": "...",
  "label": 0,
  "scenario": "education",
  "scenario_id": "A",
  "style": "report",
  "answer_type": "report",
  "domain_risk": "normal",
  "hs_domain": "",
  "domain": "education",
  "length_bucket": "200-500",
  "source_type": "public_dataset",
  "source_id": "dataset_name#row",
  "model_name": "qwen2.5-72b",
  "model_version": "2025-12",
  "prompt_id": "A2-essay-003",
  "decoding": {"temperature": 0.7, "top_p": 0.9},
  "has_sep": false,
  "split": "train"
}
```

说明:
- AI 样本必须记录 model_name/model_version/prompt_id/decoding
- 主训练集 `has_sep` 必须为 false

---

## 五、配额表（主训练集 60K）

### 5.1 场景配比（Human 与 AI 对齐）

| 场景 | 代码 | 数量 | 备注 |
| --- | --- | --- | --- |
| 学业写作 | A | 12K | 作业/实验报告/课程论文/摘要 |
| 职场写作 | B | 10K | 邮件/周报/复盘/纪要/公告 |
| 公共知识与科普问答 | C | 12K | 概念解释/问答/使用建议 |
| 社区/社媒短文本 | D | 10K | 评论/短帖/回复 |
| 消费与产品内容 | E | 10K | 商品介绍/测评/客服摘要 |
| 资讯/新闻风 | F | 6K | 资讯/公告/新闻摘要 |

### 5.2 场景内文风配比（每个场景内部约束）

- 列表要点式 + 说明书式 ≥ 35%
- 连续解释式 25%–35%
- 报告总结式 15%–25%
- 对话式 10%–20%
- 混合格式 0%–10%

### 5.3 长度分桶配比（全量）

| 长度 | 比例 |
| --- | --- |
| 80-200 | 20% |
| 200-500 | 30% |
| 500-1000 | 25% |
| 1000-2000 | 25% |

规则: <80 直接剔除。2000+ 目前不进入主训练集（可留给 OOD_HS 长文）。

### 5.4 AI 来源配比（模型维度）

- 每个模型占 AI 样本 15%–25%
- 建议池: DeepSeek / Qwen / GLM / GPT / Gemini
- 保留 1 个“整模型留出”用于 Model-OOD

---

## 六、Human 样本来源策略（按场景）

- 详见: `docs/plans/human_data_collection_2026-01-27.md`
- 关键要求: 许可明确、来源可追溯、每个场景至少两种独立来源

---

## 七、AI 样本生成策略（按场景）

- 详见: `docs/plans/ai_generation_template_framework_2026-01-27.md`
- 统一约束: 禁止“作为AI…”/拒绝/免责声明；输出真实作者式内容
- 解码多样化: temperature 0.2/0.7/1.0 分层

---

## 八、处理与采样流程

1) 统一字段与格式
2) 过滤 <80 字符、乱码、拒绝词
3) 移除 [SEP] 与混合样本
4) 精确去重 + 近重复
5) 标注场景/风格/长度分桶
6) 按配额采样生成主训练集
7) 切分 train/val/test 与 OOD（Style / Model；Security-Test 仅在需要时构建）

---

## 十、OOD_HS（高风险域）子集

目标:
- 仅用于展示跨域鲁棒性与风险边界
- 不进入主训练/验证/阈值/超参调优

默认规模:
- 总量 800（Human 400 / AI 400）
- legal 400（200/200），finance 400（200/200）

长度桶建议:
- 200-500: 20%
- 500-1000: 30%
- 1000-2000: 50%（如需长文，可从这里再分出 2000+）

必备字段:
- domain_risk=high_stakes
- hs_domain=legal|finance
- ood_split=OOD_HS

---

## 九、当前执行状态

- 目标已确认: 60K 主训练集 + Mixed-Test（Security-Test 可选）
- 待执行: 场景标签补全、配额采样（Security-Test 视需要补充）
