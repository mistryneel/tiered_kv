from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_h100_benchmark import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a RunPod H100 benchmark JSON file.")
    parser.add_argument("result_json", nargs="?", default=None)
    args = parser.parse_args()

    if args.result_json is None:
        candidates = sorted((Path(__file__).resolve().parent / "results").glob("**/*.json"))
        candidates = [p for p in candidates if not p.name.endswith("_summary.json")]
        if not candidates:
            raise SystemExit("no result JSON found")
        path = candidates[-1]
    else:
        path = Path(args.result_json)

    result = json.loads(path.read_text())
    summary = summarize(result)
    print("result", path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
