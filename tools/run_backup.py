"""Execute the notebook headless and write the pre-run backup copies.

Run:  python tools/run_backup.py

Writes
  notebooks/backup/no_show_classic_ml_prerun.ipynb   open this if the live kernel dies
  notebooks/backup/no_show_classic_ml_prerun.html    every output, charts inline, for backup slides
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "no_show_classic_ml.ipynb"
OUT_DIR = ROOT / "notebooks" / "backup"
OUT_DIR.mkdir(exist_ok=True)
STEM = "no_show_classic_ml_prerun"

base = [sys.executable, "-m", "jupyter", "nbconvert", "--execute", "--ExecutePreprocessor.timeout=600",
        str(SRC), "--output-dir", str(OUT_DIR), "--output", STEM]
subprocess.run(base + ["--to", "notebook"], check=True, cwd=ROOT / "notebooks")
subprocess.run(base + ["--to", "html"], check=True, cwd=ROOT / "notebooks")
print(f"backup written to {OUT_DIR.relative_to(ROOT)}/")
