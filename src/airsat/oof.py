"""Prediction store: every CV run is saved once and reused by stacking and analysis.

Layout: artifacts/oof/<train_dataset>/<model_key>.npz  (arrays: oof, apply__<name>)
        artifacts/oof/<train_dataset>/<model_key>.json (metrics + metadata)
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from airsat.config import OOF_DIR
from airsat.cv import CVResult

APPLY_PREFIX = "apply__"


def _paths(train_name: str, key: str, root: Path) -> tuple[Path, Path]:
    base = root / train_name / key
    return base.with_suffix(".npz"), base.with_suffix(".json")


def save_run(train_name: str, result: CVResult, meta: dict | None = None, root: Path = OOF_DIR) -> Path:
    npz_path, json_path = _paths(train_name, result.key, root)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"oof": result.oof, **{APPLY_PREFIX + k: v for k, v in result.applied.items()}}
    np.savez_compressed(npz_path, **arrays)
    record = {
        "key": result.key, "train": train_name, "cv_auc": result.cv_auc, "fold_aucs": result.fold_aucs,
        "fold_std": float(np.std(result.fold_aucs)), "applied_auc": result.applied_auc,
        "seconds": result.seconds, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(meta or {}),
    }
    json_path.write_text(json.dumps(record, indent=2))
    return npz_path


def load_run(train_name: str, key: str, root: Path = OOF_DIR) -> tuple[dict[str, np.ndarray], dict]:
    npz_path, json_path = _paths(train_name, key, root)
    with np.load(npz_path) as data:
        arrays = {name: data[name] for name in data.files}
    return arrays, json.loads(json_path.read_text())


def list_runs(train_name: str, root: Path = OOF_DIR) -> list[str]:
    folder = root / train_name
    return sorted(p.stem for p in folder.glob("*.json")) if folder.exists() else []
