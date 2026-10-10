"""Where things live on disk."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

RAW_TABLES = [
    "customers", "dda_transactions", "card_transactions", "card_statements",
    "outcomes", "interventions", "protected_attributes", "sim_truth", "sim_config",
]


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.raw = self.root / "data" / "raw"
        self.work = self.root / "data" / "work"
        self.artifacts = self.root / "artifacts"
        for d in (self.raw, self.work, self.artifacts):
            d.mkdir(parents=True, exist_ok=True)

    def write_raw(self, tables: dict[str, pd.DataFrame]) -> None:
        for name, df in tables.items():
            df.to_parquet(self.raw / f"{name}.parquet", index=False)

    def read_raw(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.raw / f"{name}.parquet")

    def write_json(self, name: str, obj) -> Path:
        path = self.artifacts / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, default=_default) + "\n")
        return path

    def read_json(self, name: str):
        return json.loads((self.artifacts / name).read_text())


def _default(o):
    import numpy as np

    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(type(o))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
