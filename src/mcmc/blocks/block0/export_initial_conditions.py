from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def export_block0_conditions(out_path: str | Path, payload: Dict[str, object]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
