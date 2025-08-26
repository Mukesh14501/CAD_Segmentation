# cadseg/utils/logging.py
from __future__ import annotations
from typing import Dict, Any
from rich.console import Console

_console = Console()

def log_kv(msg: str, kv: Dict[str, Any]):
    parts = [msg] + [f"{k}={v}" for k, v in kv.items()]
    _console.print("  ".join(parts))
