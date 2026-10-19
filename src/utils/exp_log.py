"""
exp_log.py — Append-only experiment tracking in experiments/log.csv.

Usage:
  from utils.exp_log import log_experiment
  log_experiment(model="mask_rcnn_r50", resolution=1024, val_pq=0.31, notes="baseline")
"""

import csv
import datetime as dt
from pathlib import Path
from typing import Optional

from config import CFG

COLUMNS = [
    "exp_id", "date", "model", "resolution", "preproc", "postproc",
    "val_pq", "val_sq", "val_rq", "val_dice", "lb_score", "sec_per_img", "notes",
]


def init_log(path: Optional[Path] = None) -> Path:
    """Create the log with its header if it does not exist yet."""
    path = Path(path or CFG.paths.EXPERIMENT_LOG)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(COLUMNS)
    return path


def log_experiment(path: Optional[Path] = None, **fields) -> str:
    """Append one run. Unknown keys raise; missing keys are left blank. Returns exp_id."""
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise KeyError(f"Unknown log columns: {sorted(unknown)}")
    path = init_log(path)
    with open(path, newline="") as f:
        n_rows = sum(1 for _ in csv.reader(f)) - 1
    fields.setdefault("exp_id", f"exp{n_rows + 1:03d}")
    fields.setdefault("date", dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    row = []
    for col in COLUMNS:
        v = fields.get(col, "")
        row.append(f"{v:.4f}" if isinstance(v, float) else v)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)
    print(f"[ExpLog] Logged {fields['exp_id']} -> {path}")
    return fields["exp_id"]
