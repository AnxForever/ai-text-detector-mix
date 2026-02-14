param(
    [string]$V11APath = "models/bert_v11a",
    [string]$V11BPath = "models/bert_v11b",
    [string]$V11CPath = "",
    [string]$OutJson = "docs/plans/eval_comparison_v10_v11ab.json"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/4] Fair-test evaluation for V10/V11a/V11b"
$models = @(
  "bert_v10_augmented=models/bert_v10_augmented",
  "bert_v11a=$V11APath",
  "bert_v11b=$V11BPath"
)
if ($V11CPath -and $V11CPath.Trim().Length -gt 0) {
  $models += "bert_v11c=$V11CPath"
}

py -3 scripts/evaluation/eval_fair_models_for_gate.py --models $models --output-json $OutJson

Write-Host "[2/4] Regression gate: V11a vs V10"
py -3 scripts/evaluation/check_v11_regression_gate.py `
  --baseline-json $OutJson `
  --candidate-json $OutJson `
  --baseline-key bert_v10_augmented `
  --candidate-key bert_v11a `
  --output-json docs/plans/v11a_regression_gate.json `
  --output-md docs/plans/v11a_regression_gate.md

Write-Host "[3/4] Regression gate: V11b vs V10"
py -3 scripts/evaluation/check_v11_regression_gate.py `
  --baseline-json $OutJson `
  --candidate-json $OutJson `
  --baseline-key bert_v10_augmented `
  --candidate-key bert_v11b `
  --output-json docs/plans/v11b_regression_gate.json `
  --output-md docs/plans/v11b_regression_gate.md

Write-Host "[4/4] Calibration (mandatory)"
py -3 scripts/evaluation/calibrate_temperature.py --model-path $V11APath
py -3 scripts/evaluation/calibrate_temperature.py --model-path $V11BPath
if ($V11CPath -and $V11CPath.Trim().Length -gt 0) {
  py -3 scripts/evaluation/calibrate_temperature.py --model-path $V11CPath
}

Write-Host "[DONE] Outputs:"
Write-Host "  $OutJson"
Write-Host "  docs/plans/v11a_regression_gate.json"
Write-Host "  docs/plans/v11b_regression_gate.json"
