# 答辩统一修复清单 (2026-02-12)

## 已完成

- [x] API 分类模型默认切换到 `models/bert_v10_augmented`
- [x] API 增加检测模型环境变量 (`DETECTOR_*`)
- [x] 移除聊天接口硬编码默认密钥，改为强制读取 `OPENAI_API_KEY`
- [x] 前端检测与聊天接口统一使用 `NEXT_PUBLIC_API_URL`
- [x] 前端移除默认测试 API Key
- [x] 新增答辩口径快照文档 `DEFENSE_CURRENT_STATUS.md`
- [x] 根文档 `README.md` 同步为 V10 口径并更新目录结构
- [x] 根 `CLAUDE.md` 同步推荐模型/指标/入口文档
- [x] `datasets/CLAUDE.md` 同步 core_v1/core_v2/core_v3/merged_v2 规模
- [x] `datasets/registry.json` 更新为 2026-02-12，并补充 `core_v2/core_v3/merged_v2`
- [x] `api/CLAUDE.md`、`scripts/CLAUDE.md`、`docs/CLAUDE.md`、`DOCS_INDEX.md` 同步更新
- [x] `FINAL_RESULTS.md` 标记为基线阶段并指向最新答辩口径

## 验证结果

- [x] `py -3 -m py_compile api/api.py`
- [x] `py -3 -c "import json; ..."` 校验 `datasets/registry.json` 格式
- [x] `pnpm -C frontend exec tsc --noEmit`
- [ ] `pnpm -C frontend lint`（当前环境缺少 eslint 命令）
