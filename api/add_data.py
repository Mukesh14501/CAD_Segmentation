from __future__ import annotations
import io
import zipfile
import tempfile
import subprocess
import sys
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from cadseg.dataio.ingest import ingest_week

app = FastAPI(title="CADSeg Admin API", version="1.1")

def _spawn_training(run_dir: str = "runs/prod", configs_dir: str = "configs", limit_valid: int = 0) -> int:
    """
    Spawn training/fine-tuning as a separate process (non-blocking).
    Uses the stable run_dir so future runs fine-tune on top of existing best.pth.
    """
    py = sys.executable
    cmd = [
        py, "-m", "cadseg.cli.train",
        "--configs", configs_dir,
        "--run_dir", run_dir,
    ]
    if limit_valid > 0:
        cmd += ["--limit_valid", str(limit_valid)]

    # Make sure albumentations doesn't try to call home in corp networks
    env = os.environ.copy()
    env.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    # Non-blocking spawn
    proc = subprocess.Popen(cmd, env=env)
    return proc.pid


@app.post("/ingest")
async def ingest_endpoint(
    data_zip: UploadFile = File(..., description="ZIP containing images/ and masks/"),
    data_root: str = Form("data"),                    # canonical dataset root (has images/, masks/, splits/)
    seed: int = Form(42),
    train_ratio: float = Form(0.8),                   # 80/20 as requested
    run_dir: str = Form("runs/prod"),                 # stable run dir so it fine-tunes
    configs_dir: str = Form("configs"),
    limit_valid: int = Form(0),
):
    """
    1) Extract the uploaded ZIP into a temp dir.
    2) Ingest it into canonical data tree
       - images appended
       - masks appended per class; new class folders auto-created
       - meta/latest_id.txt updated
       - splits: +80% new to train, +20% new to valid (valid = previous valid + new 20%)
       - split files deduped
    3) Trigger training/fine-tuning (non-blocking) on the stable run_dir.
    """
    raw = await data_zip.read()
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        ext_dir = td_path / "new_data"
        # extract from bytes directly
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            zf.extractall(ext_dir)

        summary = ingest_week(Path(data_root), ext_dir, seed=seed, train_ratio=train_ratio)

    # pid = _spawn_training(run_dir=run_dir, configs_dir=configs_dir, limit_valid=limit_valid)

    return JSONResponse({
        "ingest_summary": summary,
        # "train_trigger": {"run_dir": run_dir, "pid": pid}
    })

# Example run:
# uvicorn api.add_data:app --host 127.0.0.1 --port 8001 --reload
