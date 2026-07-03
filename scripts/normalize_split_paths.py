"""Rewrites data_splits/*.csv so image_path/mask_path are relative to paths.melanoma_data,
instead of machine-specific absolute paths. This is what makes the splits portable across
machines: anyone can clone the repo, set MELANOMA_DATA_DIR (or edit configs/base.yaml) to
point at their own copy of the raw data, and the existing splits just work.

Run this once after generating new splits that still have absolute paths baked in. Safe to
re-run; rows that are already relative (or absolute but outside the data root) are left as-is.

Usage: python scripts/normalize_split_paths.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils.config import load_config

_COLUMNS = ("image_path", "mask_path", "cached_crop_path")


def _to_relative(p, root: Path) -> str:
    if not isinstance(p, str) or not p:
        return p
    path = Path(p)
    if not path.is_absolute():
        return p
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return p  # not under root — leave it, resolve_dataset_paths() will pass it through


def main() -> None:
    cfg = load_config()
    root = Path(cfg.paths.melanoma_data)
    splits_dir = Path(cfg.paths.data_splits)

    print(f"  Data root   : {root}")
    print(f"  Splits dir  : {splits_dir}\n")

    for csv_path in sorted(splits_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        changed_cols = []
        for col in _COLUMNS:
            if col not in df.columns:
                continue
            new_col = df[col].apply(lambda p: _to_relative(p, root))
            if not new_col.equals(df[col]):
                df[col] = new_col
                changed_cols.append(col)
        if changed_cols:
            df.to_csv(csv_path, index=False)
            print(f"  {csv_path.name:<20} rewrote {', '.join(changed_cols)}")
        else:
            print(f"  {csv_path.name:<20} already relative, no change")


if __name__ == "__main__":
    main()
