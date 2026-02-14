# V11 Post-Training Runbook

Use this runbook after `bert_v11a` and `bert_v11b` training finishes.
If boundary drift appears on long text, include `bert_v11c`.

## 1) Fair-test evaluation JSON (gate-compatible)

```bash
py -3 scripts/evaluation/eval_fair_models_for_gate.py \
  --models bert_v10_augmented bert_v11a bert_v11b bert_v11c \
  --output-json docs/plans/eval_comparison_v10_v11ab.json
```

If your model folder names are different, use explicit mapping:

```bash
py -3 scripts/evaluation/eval_fair_models_for_gate.py \
  --models bert_v10_augmented=models/bert_v10_augmented \
           bert_v11a=models/<actual_v11a_dir> \
           bert_v11b=models/<actual_v11b_dir> \
           bert_v11c=models/<actual_v11c_dir> \
  --output-json docs/plans/eval_comparison_v10_v11ab.json
```

## 2) Regression gate (V11a vs V10)

```bash
py -3 scripts/evaluation/check_v11_regression_gate.py \
  --baseline-json docs/plans/eval_comparison_v10_v11ab.json \
  --candidate-json docs/plans/eval_comparison_v10_v11ab.json \
  --baseline-key bert_v10_augmented \
  --candidate-key bert_v11a \
  --output-json docs/plans/v11a_regression_gate.json \
  --output-md docs/plans/v11a_regression_gate.md
```

## 3) Regression gate (V11b vs V10)

```bash
py -3 scripts/evaluation/check_v11_regression_gate.py \
  --baseline-json docs/plans/eval_comparison_v10_v11ab.json \
  --candidate-json docs/plans/eval_comparison_v10_v11ab.json \
  --baseline-key bert_v10_augmented \
  --candidate-key bert_v11b \
  --output-json docs/plans/v11b_regression_gate.json \
  --output-md docs/plans/v11b_regression_gate.md
```

## 4) Calibration (mandatory for each V11 model)

```bash
py -3 scripts/evaluation/calibrate_temperature.py --model-path models/<actual_v11a_dir>
py -3 scripts/evaluation/calibrate_temperature.py --model-path models/<actual_v11b_dir>
py -3 scripts/evaluation/calibrate_temperature.py --model-path models/<actual_v11c_dir>
```

## One-shot PowerShell pipeline (optional)

```powershell
pwsh scripts/evaluation/run_v11ab_posttrain.ps1 `
  -V11APath models/<actual_v11a_dir> `
  -V11BPath models/<actual_v11b_dir>
```

## Decision rule

- If three-set average degradation is `> 0.5`, rollback to V10.
- Prefer V11b only if it passes gate and improves weak-domain behavior.
