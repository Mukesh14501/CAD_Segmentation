# cadseg/api/admin.py
from __future__ import annotations
import io
import zipfile
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from cadseg.dataio.ingest import ingest_week

app = FastAPI(title="CADSeg Admin API", version="1.0")

@app.post("/ingest")
async def ingest_endpoint(
    data_zip: UploadFile = File(..., description="ZIP containing images/ and masks/"),
    data_root: str = Form("data"),
    seed: int = Form(42),
    train_ratio: float = Form(0.9),
):
    # Save uploaded zip to temp, extract, then call ingest_week
    raw = await data_zip.read()
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        zpath = td_path / "new_data.zip"
        zpath.write_bytes(raw)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            zf.extractall(td_path / "new_data")

        summary = ingest_week(Path(data_root), td_path / "new_data", seed=seed, train_ratio=train_ratio)

    return JSONResponse(summary)



# uvicorn api.add_data:app --host 0.0.0.0 --port 8001