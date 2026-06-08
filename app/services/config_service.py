from __future__ import annotations

import json
from pathlib import Path


class ConfigService:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = {}

    def load(self) -> dict:
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        return self.data

    def save(self, payload: dict) -> None:
        self.data = payload
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
