# 中文AI文本检测系统 - 答辩口径快照 (2026-02-13)

## 1) 当前推荐配置

- 分类模型: `models/bert_v11c_boundary_fix`
- 边界模型: `models/bert_span_detector`
- API 默认端口: `http://localhost:8000`
- 前端默认读取: `NEXT_PUBLIC_API_URL`（未配置时回退 `http://localhost:8000`）

## 2) 关键性能指标 (V11c)

数据来源:
- `models/bert_v11c_boundary_fix/eval_comparison.json`
- `models/bert_v11c_boundary_fix/training_log.json`
- `docs/plans/v11_four_way_comparison.json`

| 指标 | 数值 |
|------|------|
| 验证集准确率 | 98.75% |
| 独立评估集准确率 (910) | 98.57% |
| 三集平均准确率 | 98.56% |
| merged_v2_val_clean | 99.13% |
| Token 级边界检测准确率 | 96.69% |

补充:
- 训练样本数: 63,113
- V11c 数据策略: V10 经风险治理（移除模板/unknown）+ 弱域增补 + 长文AI边界修复
- 最优温度缩放: `T=0.8165`
- ECE (校准误差): 0.0034

### V11c vs V10 改进

| 指标 | V10 | V11c | 变化 |
|------|-----|------|------|
| 独立评估集准确率 | 97.69% | 98.57% | **+0.88%** |
| 三集平均 | 98.36% | 98.56% | **+0.20%** |
| independent 总错误 | 21 | 13 | **-38%** |
| LLaMA-405B 检出率 | 88.9% | 100% | **+11.1%** |
| formal_collected 正确率 | 96.0% | 96.5% | **+0.5%** |

## 3) 数据集口径

统计口径为当前仓库实际文件:

| 数据集 | 切分 | 样本数 |
|--------|------|--------|
| core_v1 | train/val/test | 46,849 / 5,856 / 5,858 |
| core_v2 | train/val/test | 57,435 / 7,179 / 7,180 |
| core_v3 | train/val | 61,045 / 6,783 |
| merged_v2 | train/val | 61,872 / 7,475 |
| merged_v2 (v10) | train_v10 | 62,980 |
| merged_v2 (v11c) | train_v11c_candidate | 63,187 |

## 4) 答辩演示建议

1. 后端启动:

```bash
py -3 api/api.py
```

2. 前端启动:

```bash
cd frontend
pnpm dev
```

3. 演示接口:
- 文本检测: `POST /api/detect`
- 续写/润色: `POST /v1/chat/completions`

## 5) 文档说明

- `docs/project/FINAL_RESULTS.md` 主要记录早期基线阶段（v2）。
- `docs/plans/v11_four_way_comparison.md` 记录 V10/V11a/V11b/V11c 四模型对比。
- 本文件是 2026-02-13 的最新答辩口径快照，优先用于汇报与答辩。
