#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m pip install -q -r requirements-runpod.txt

python - <<'PY'
import importlib
import sys
from pathlib import Path

root = Path.cwd().resolve().parent
sys.path.insert(0, str(root))

missing = []
for name in ("torch", "transformers", "pandas", "matplotlib"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append((name, repr(exc)))

if not (root / "tiered_kv").exists():
    missing.append(("tiered_kv", f"expected sibling folder at {root / 'tiered_kv'}"))

if missing:
    for name, err in missing:
        print(f"missing {name}: {err}")
    raise SystemExit(1)

import torch
import transformers

print("setup ok")
print("python", sys.version)
print("transformers", transformers.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
