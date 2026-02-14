$log = "models\v10_train.log"
while (-not (Test-Path $log)) { Start-Sleep -Seconds 10 }
while (-not (Select-String -Path $log -Pattern "训练完成" -Quiet)) { Start-Sleep -Seconds 30 }
$env:PYTHONIOENCODING = "utf-8"
$OutputEncoding = [System.Text.Encoding]::UTF8
py -3.12 scripts\evaluation\eval_v10_compare.py > models\v10_eval_compare.log 2>&1
