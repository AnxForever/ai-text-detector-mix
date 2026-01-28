"""Plan data fill pipeline steps from a JSON config.

This script only creates a plan (JSON/MD). It does not call any APIs or
generate data. Use --execute to write the plan files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List


os.environ.setdefault("PYTHONIOENCODING", "utf-8")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)


REQUIRED_CONFIG_KEYS = ["run_name", "output_root", "targets"]


def load_config(path: str) -> Dict[str, Any]:
    """Load JSON config from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_config(config: Dict[str, Any]) -> None:
    """Validate required fields in config."""
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")
    if not isinstance(config["targets"], list) or not config["targets"]:
        raise ValueError("Config 'targets' must be a non-empty list.")


def build_run_dir(output_root: str, run_name: str) -> str:
    """Build run output directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_root, f"{run_name}_{timestamp}")


def build_tasks(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert target definitions into task list."""
    tasks = []
    ai_enabled = config.get("ai_generation", {}).get("enabled", False)
    human_enabled = config.get("human_collection", {}).get("enabled", False)

    for target in config["targets"]:
        label = target.get("label", "UNKNOWN")
        action = "collect_human" if label == "HUMAN" else "generate_ai"
        blocked = False
        if label == "HUMAN" and not human_enabled:
            blocked = True
        if label == "AI" and not ai_enabled:
            blocked = True

        tasks.append(
            {
                "action": action,
                "label": label,
                "style": target.get("style", ""),
                "domain": target.get("domain", ""),
                "length_bucket": target.get("length_bucket", ""),
                "target_count": target.get("target_count", 0),
                "priority": target.get("priority", ""),
                "notes": target.get("notes", ""),
                "blocked": blocked,
            }
        )
    return tasks


def write_plan_files(
    run_dir: str, config: Dict[str, Any], tasks: List[Dict[str, Any]]
) -> None:
    """Write plan JSON and Markdown summary."""
    os.makedirs(run_dir, exist_ok=True)
    plan_json_path = os.path.join(run_dir, "plan.json")
    plan_md_path = os.path.join(run_dir, "plan.md")

    with open(plan_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": config["run_name"],
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config_path": config.get("config_path", ""),
                "tasks": tasks,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(plan_md_path, "w", encoding="utf-8") as f:
        f.write("# Data Fill Plan\n\n")
        f.write(f"- run_name: {config['run_name']}\n")
        f.write(
            f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write("\n## Tasks\n\n")
        f.write("| action | label | style | domain | length | target | priority | blocked |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for task in tasks:
            f.write(
                f"| {task['action']} | {task['label']} | {task['style']} "
                f"| {task['domain']} | {task['length_bucket']} | "
                f"{task['target_count']} | {task['priority']} | "
                f"{task['blocked']} |\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create data fill plan from config.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Override output_root in config (optional).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write plan files to disk (default is dry-run).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config["config_path"] = args.config
    validate_config(config)

    output_root = args.output_dir or config["output_root"]
    run_dir = build_run_dir(output_root, config["run_name"])
    tasks = build_tasks(config)

    print("Plan summary:")
    print(f"- run_name: {config['run_name']}")
    print(f"- output_root: {output_root}")
    print(f"- tasks: {len(tasks)}")

    blocked_count = sum(1 for task in tasks if task["blocked"])
    print(f"- blocked tasks: {blocked_count}")

    if args.execute:
        write_plan_files(run_dir, config, tasks)
        print(f"Plan written to: {run_dir}")
    else:
        print("Dry-run only. Use --execute to write plan files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
