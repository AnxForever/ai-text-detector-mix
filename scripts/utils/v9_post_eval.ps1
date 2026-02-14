$log = "models\v9_train.log"
while (-not (Test-Path $log)) { Start-Sleep -Seconds 10 }
while (-not (Select-String -Path $log -Pattern "训练完成" -Quiet)) { Start-Sleep -Seconds 30 }
$env:PYTHONIOENCODING = "utf-8"
$OutputEncoding = [System.Text.Encoding]::UTF8
py -3.12 scripts\evaluation\eval_sliced.py --models bert_v9_p0_supplement > models\v9_eval_sliced.log 2>&1
py -3.12 scripts\evaluation\calibrate_temperature.py --model bert_v9_p0_supplement > models\v9_calibrate.log 2>&1
