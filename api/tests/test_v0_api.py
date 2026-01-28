#!/usr/bin/env python3
"""Simple API configuration sanity check.

This script verifies that api/API_KEYS.md exists and prints guidance.
It does not make network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Entry point."""
    root = Path(__file__).resolve().parents[2]
    api_keys = root / "api" / "API_KEYS.md"
    if not api_keys.exists():
        print(f"API_KEYS.md not found at: {api_keys}")
        sys.exit(1)

    print(f"API_KEYS.md found at: {api_keys}")
    print("If API calls fail, verify keys and endpoint configuration.")


if __name__ == "__main__":
    main()
