# OOD_HS 高风险域测试子集方案

> 生成时间: 2026-01-28
> 目标: 法律/金融高风险域独立 OOD 测试子集（仅展示鲁棒性与风险边界，不进入训练/调参）。

---

## 1. 范围与原则

- 仅覆盖 legal / finance
- 不进主训练、不参与配额统计、不用于阈值/超参调优
- 标注为 `domain_risk=high_stakes`，并记录 `hs_domain` 与 `ood_split=OOD_HS`

---

## 2. 高风险域判定规则（满足任一）

### legal
- 法规条款解读、合规通知、隐私/数据保护、合同条款、责任界定、申诉/仲裁、处罚/罚则

### finance
- 资产配置、收益率/回撤、投资建议、理财产品条款、杠杆/保证金、风控指标

---

## 3. 必备字段（最小实现）

- scenario_id: A–F（强制重映射）
- answer_type: list / guide / report / explanation / dialogue / mixed
- domain_risk: high_stakes
- hs_domain: legal / finance
- ood_split: OOD_HS

---

## 4. 规模与组成（默认配置）

- 总量: 800
- 标签: Human 400 / AI 400
- 子域: legal 400（200/200），finance 400（200/200）

### 长度桶（默认）
- 200-500: 20%
- 500-1000: 30%
- 1000-2000: 40%–50%
- 2000+: 10%–20%（仅用于长文鲁棒性展示，可选）

---

## 5. 生成与采集原则

### AI 侧
- 模型池均衡: DeepSeek / Qwen / GLM / GPT / Gemini
- MUST 仅保留 4 件套: 禁词 + 长度 + 结构下限 + 至少 1 个阈值/数字
- 其余细节放 NICE，避免过度模板化

### Human 侧
- 法律: 法规条文、合规指南、隐私政策、公开裁判文书摘要（客观叙述）
- 金融: 投教材料、产品说明书条款、风险揭示书、财经新闻摘要、研究报告客观段落
- Human 也需覆盖 list/report/explanation 三类结构，避免结构偏差

---

## 6. 评测输出（答辩可直接展示）

- OOD_HS 总体指标（Accuracy/F1）
- legal vs finance 子域对比
- answer_type 分桶指标（list/report/explanation 为主）

---

## 7. 场景重映射规则（最小可执行）

- 合规通知/内部制度/流程规范: 归 B（职场写作）
- 对外公告/通报/事件说明: 归 F（资讯/新闻风）
- 投资方法科普/风险提示: 归 C（公共知识）
- 产品公告/市场资讯: 归 F（资讯/新闻风）

---

## 8. 数据输出建议

- 数据集位置: datasets/eval/ood_high_stakes/
- 产出文件: ood_hs.jsonl / ood_hs_stats.json
