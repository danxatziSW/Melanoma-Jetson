import os
import threading
from pathlib import Path
from typing import Any

import pandas as pd

_locks: dict[str, threading.Lock] = {}
_locks_mutex = threading.Lock()


def resolve_dataset_paths(
    df: pd.DataFrame,
    root: str | Path,
    columns: tuple[str, ...] = ("image_path", "mask_path", "cached_crop_path"),
) -> pd.DataFrame:
    """Joins relative path columns onto `root` so split CSVs work on any machine.

    Paths already absolute (e.g. legacy CSVs) are left untouched, and empty/NaN
    values (like `mask_path` when there's no mask) pass through unchanged.
    """
    root = Path(root)
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(
            lambda p: str(root / p) if isinstance(p, str) and p and not Path(p).is_absolute() else p
        )
    return df


def _get_lock(filepath: str) -> threading.Lock:
    with _locks_mutex:
        if filepath not in _locks:
            _locks[filepath] = threading.Lock()
        return _locks[filepath]


def write_excel_sheet(filepath: str | Path, sheet_name: str, data: Any) -> None:
    filepath = str(filepath)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    lock = _get_lock(filepath)

    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    with lock:
        if Path(filepath).exists():
            with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(filepath, engine="openpyxl", mode="w") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)


def append_excel_row(filepath: str | Path, sheet_name: str, row: dict) -> None:
    filepath = str(filepath)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    lock = _get_lock(filepath)

    with lock:
        if Path(filepath).exists():
            try:
                existing = pd.read_excel(filepath, sheet_name=sheet_name)
                updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            except Exception:
                updated = pd.DataFrame([row])
            with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                updated.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            df = pd.DataFrame([row])
            with pd.ExcelWriter(filepath, engine="openpyxl", mode="w") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
