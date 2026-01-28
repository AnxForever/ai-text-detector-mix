import os
from pathlib import Path

def main():
    root = Path(r"c:\datacollection")
    file_path = root / "scripts" / "evaluation" / "predict_probabilities.py"
    if not file_path.exists():
        print(f"Missing: {file_path}")
        return

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("Note: To limit CPU usage, set env vars: OMP_NUM_THREADS=2, MKL_NUM_THREADS=2")
    print("GPU available is False in current environment.")

if __name__ == "__main__":
    main()
