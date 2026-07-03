import copy
import os
import types
from pathlib import Path

import yaml


def _resolve_paths(cfg: dict, project_root: Path) -> dict:
    """Anchors relative paths.* entries to the repo root; melanoma_data can be
    overridden with the MELANOMA_DATA_DIR env var since it lives outside the repo."""
    paths = cfg.get("paths")
    if not paths:
        return cfg
    resolved = dict(paths)
    for key in ("data_splits", "outputs"):
        if key in resolved:
            p = Path(resolved[key])
            resolved[key] = str(p if p.is_absolute() else (project_root / p).resolve())
    if "melanoma_data" in resolved:
        resolved["melanoma_data"] = os.environ.get("MELANOMA_DATA_DIR", resolved["melanoma_data"])
    cfg = dict(cfg)
    cfg["paths"] = resolved
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _dict_to_namespace(d: dict) -> types.SimpleNamespace:
    ns = types.SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_namespace(v) if isinstance(v, dict) else v)
    return ns


def load_config(model_name: str | None = None, config_path: str | None = None) -> types.SimpleNamespace:
    root = Path(__file__).resolve().parents[2] / "configs"
    base_cfg = yaml.safe_load((root / "base.yaml").read_text())

    if config_path is not None:
        override = yaml.safe_load(Path(config_path).read_text())
        merged = _deep_merge(base_cfg, override)
    elif model_name is not None:
        model_cfg_path = root / "models" / f"{model_name}.yaml"
        override = yaml.safe_load(model_cfg_path.read_text())
        merged = _deep_merge(base_cfg, override)
    else:
        merged = base_cfg

    merged = _resolve_paths(merged, root.parent)
    return _dict_to_namespace(merged)


def load_segmentation_config() -> types.SimpleNamespace:
    root = Path(__file__).resolve().parents[2] / "configs"
    base_cfg = yaml.safe_load((root / "base.yaml").read_text())
    seg_cfg = yaml.safe_load((root / "segmentation.yaml").read_text())
    merged = _resolve_paths(_deep_merge(base_cfg, seg_cfg), root.parent)
    return _dict_to_namespace(merged)


def load_ensemble_config() -> types.SimpleNamespace:
    root = Path(__file__).resolve().parents[2] / "configs"
    cfg = yaml.safe_load((root / "ensemble.yaml").read_text())
    return _dict_to_namespace(_resolve_paths(cfg, root.parent))
