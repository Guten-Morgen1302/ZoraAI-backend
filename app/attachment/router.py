from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas import AttachmentAnalyzeResponse, AttachmentEngineResult

router = APIRouter(prefix="/attachment", tags=["attachment-analysis"])

_SANDBOX_ROOT = Path(__file__).resolve().parent.parent / "attachment-sandbox"


def _ensure_sandbox_path() -> None:
    sandbox_root_str = str(_SANDBOX_ROOT)
    if sandbox_root_str not in sys.path:
        sys.path.insert(0, sandbox_root_str)


def _load_pipeline_runner():
    _ensure_sandbox_path()
    try:
        pipeline_module = importlib.import_module("app.static_analysis.pipeline")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unable to import attachment static pipeline: {exc}") from exc

    run_static_pipeline = getattr(pipeline_module, "run_static_pipeline", None)
    if run_static_pipeline is None or not callable(run_static_pipeline):
        raise RuntimeError("Attachment static pipeline entrypoint is missing")

    return run_static_pipeline


def _normalize_engine_results(engines: Any) -> dict[str, AttachmentEngineResult]:
    if not isinstance(engines, dict):
        return {}

    result: dict[str, AttachmentEngineResult] = {}
    for engine_name, payload in engines.items():
        if not isinstance(payload, dict):
            continue

        result[str(engine_name)] = AttachmentEngineResult(
            is_flagged=bool(payload.get("is_flagged", False)),
            hits=payload.get("hits") if isinstance(payload.get("hits"), list) else None,
            signature=(str(payload.get("signature")) if payload.get("signature") is not None else None),
            score=(float(payload.get("score")) if payload.get("score") is not None else None),
        )

    return result


@router.post("/analyze", response_model=AttachmentAnalyzeResponse, status_code=status.HTTP_200_OK)
async def analyze_attachment(file: UploadFile | None = File(default=None)):
    if file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")

    filename = os.path.basename(file.filename or "uploaded_attachment")
    if not filename:
        filename = "uploaded_attachment"

    suffix = Path(filename).suffix
    temp_file_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            if not content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file is empty",
                )
            tmp.write(content)
            temp_file_path = tmp.name

        run_static_pipeline = _load_pipeline_runner()
        report = run_static_pipeline(temp_file_path)

        if not isinstance(report, dict):
            raise RuntimeError("Attachment pipeline returned an invalid response")

        return AttachmentAnalyzeResponse(
            filename=filename,
            file_size=len(content),
            final_verdict=str(report.get("final_verdict") or "unknown"),
            engines=_normalize_engine_results(report.get("engines")),
            features=report.get("features") if isinstance(report.get("features"), dict) else {},
        )
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Attachment analysis failed: {exc}",
        ) from exc
    finally:
        await file.close()
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass
