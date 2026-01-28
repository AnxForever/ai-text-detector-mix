# 方案决策口径（本科毕设可控版）

> 生成时间: 2026-01-28
> 目标: 保证可实现、成本可控、答辩表述清晰。

---

## 1) 场景 × 回答类型：双轴结构（场景为主轴）

- 主轴: scenario(A–F)
- 副轴: answer_type（不新增枚举，复用生成脚本支持 6 类）
  - list / guide / report / explanation / dialogue / mixed

### 自动映射规则（强信号优先）
优先级：dialogue > guide > report > list > mixed > explanation

- dialogue: 出现“问/答、Q/A、用户/助手、多轮引用、对话分段”等强特征
- guide: “步骤1/2/3、首先/然后/最后、点击/输入/执行/安装/配置”等动作序列密集
- report: “背景/目标/范围/数据/结果/结论/复盘/行动项/负责人/截止/影响范围/发布/公告”等骨架字段（命中≥2）
- list: 条目符号密集（行首 - • 1. 等占比高）或“必须/不得/阈值/参数：值/处罚”等规则语气密集
- mixed: 同时强命中 list/guide + explanation/report
- explanation: 以连续段落解释为主（定义/因果/对比/例子），且不强命中以上类型

---

## 2) spec 文风处理策略（不改代码，低成本）

- 不新增 style_plan=spec
- 保留 spec_like=1 标记
- spec 映射规则:
  - 条款/规范/参数约束/处罚制度 → list（spec_like=1）
  - FAQ/使用说明/操作规范偏流程 → guide（spec_like=1）
  - 公告/制度发布/变更说明/复盘通报 → report（spec_like=1）

---

## 3) 长度策略（主训练不保留 2000+）

- 主训练长度桶: 80–200 / 200–500 / 500–1000 / 1000–2000
- 2000+ 不进入主训练
- OOD_HS 可保留少量 2000+（展示长文鲁棒性）

---

## 4) MUST / NICE 比例（降低拒收率）

- MUST 约 30%，NICE 约 70%
- 每条模板 MUST 建议“4 件套”:
  1. 禁止自我指代/拒绝免责声明
  2. 长度范围
  3. 结构下限（条目≥4 或 步骤≥4 或 Q/A≥3）
  4. 至少 1 个阈值/参数/可核验数字

---

## 5) D 场景补齐短文本

- 必补:
  - 80–200 普通短评论（非楼中楼）
  - 200–500 简短回复式问答（2–3 轮）

---

## 6) 配额策略（先场景后类型）

- 采样方式: P(scenario) × P(answer_type | scenario)
- 默认条件分布:
  - A 学业: explanation 35%，report 30%，list 20%，guide 10%，mixed 5%，dialogue 0%
  - B 职场: report 45%，list 30%，guide 15%，mixed 5%，explanation 5%，dialogue 0%
  - C 科普: explanation 55%，guide 20%，list 15%，dialogue 5%，mixed 5%，report 0%
  - D 社媒: dialogue 35%，explanation 20%，mixed 15%，list 15%，guide 15%，report 0%
  - E 消费: report 35%，list 30%，mixed 15%，explanation 15%，dialogue 5%，guide 0%
  - F 资讯: report 55%，explanation 25%，list 15%，mixed 5%，guide 0%，dialogue 0%

- 全局约束: spec_like 占比 20%–30%（修复列表/说明书式盲区）

---

## 7) Manus 模板融合原则（强制重映射）

- 强制回到 A–F（不允许改含义，不引入 G/H）
- 高风险域（法律/金融）只进 OOD_HS
- spec 一律映射为 list/guide/report，保留 spec_like=1
- 高风险模板加元数据:
  - domain_risk=high_stakes
  - hs_domain=legal/finance
  - ood_split=OOD_HS
