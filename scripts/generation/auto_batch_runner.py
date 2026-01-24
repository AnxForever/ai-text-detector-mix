import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


OUTPUT_DIR = Path("new_plan_datasets")
PLAN_FILE = OUTPUT_DIR / "auto_generation_plan_v2_followup.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "parallel_dataset.csv"
RUN_LOG = Path("auto_batch_runner.log")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))
TARGET_TOTAL = int(os.getenv("TARGET_TOTAL", "10000"))
SLEEP_BETWEEN_BATCHES = int(os.getenv("SLEEP_BETWEEN_BATCHES", "15"))


def count_generated_parallel(plan_path: Path) -> int:
    if not plan_path.exists():
        return 0
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    return sum(1 for item in data if item.get("generated_parallel"))


def count_csv_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = sum(1 for _ in reader)
    return max(0, rows - 1)


def log_line(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_batch(env: dict, output_path: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = Path(f"parallel_generation_{timestamp}.stdout.log")
    stderr_path = Path(f"parallel_generation_{timestamp}.stderr.log")
    cmd = [env.get("PYTHON_EXECUTABLE", "py"), "scripts/generation/parallel_generation.py"]

    log_line(f"Starting batch: size={env['TARGET_SAMPLES']} stdout={stdout_path} stderr={stderr_path}")
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr, check=False)
    log_line(f"Batch finished with code={result.returncode} output={output_path}")


def main():
    output_path = Path(os.getenv("PARALLEL_OUTPUT_FILE", str(DEFAULT_OUTPUT)))

    base_env = os.environ.copy()
    base_env["TARGET_SAMPLES"] = str(BATCH_SIZE)
    base_env["SAVE_INTERVAL"] = os.getenv("SAVE_INTERVAL", "10")
    base_env["MAX_WORKERS"] = os.getenv("MAX_WORKERS", "3")
    base_env["REQUEST_TIMEOUT"] = os.getenv("REQUEST_TIMEOUT", "60")
    base_env["PARALLEL_OUTPUT_FILE"] = str(output_path)
    base_env["RATE_LIMIT_SLEEP"] = os.getenv("RATE_LIMIT_SLEEP", "10")
    base_env["ERROR_BACKOFF_SLEEP"] = os.getenv("ERROR_BACKOFF_SLEEP", "3")
    base_env["MAX_RETRIES"] = os.getenv("MAX_RETRIES", "4")

    log_line(f"Runner started: target_total={TARGET_TOTAL} batch_size={BATCH_SIZE}")

    while True:
        completed = count_generated_parallel(PLAN_FILE)
        if completed >= TARGET_TOTAL:
            log_line(f"Target reached: generated_parallel={completed}")
            break

        rows = count_csv_rows(output_path)
        log_line(f"Progress: generated_parallel={completed} csv_rows={rows}")

        run_batch(base_env, output_path)

        # cooldown to avoid rate limiting
        if SLEEP_BETWEEN_BATCHES > 0:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    log_line("Runner completed.")


if __name__ == "__main__":
    main()
