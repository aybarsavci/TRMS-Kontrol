"""JSON persistence of tuned controller parameters.

Each controller entry stores a SHA-256 signature of everything that affects the
tuning result (model parameters, sample time, tuning scenario, objective,
controller settings and search budget). A cached entry is reused only if the
signature still matches; otherwise the controller is re-tuned automatically.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ALGO_VERSION = "1.0"


def signature(payload: dict) -> str:
    blob = json.dumps({"algo_version": ALGO_VERSION, **payload}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class ParameterCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                print(f"[cache] {self.path} is unreadable; ignoring it.")

    def get(self, key: str, sig: str):
        entry = self.data.get(key)
        if entry and entry.get("signature") == sig:
            return entry
        return None

    def put(self, key: str, sig: str, params: dict, report: dict):
        self.data[key] = {
            "signature": sig,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "params": params,
            "tuning_report": report,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=False))
