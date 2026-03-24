from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import boto3
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AttachmentAnalysis, AttachmentRequest
from app.schemas import AttachmentAnalyzeResponse, AttachmentEngineResult

router = APIRouter(prefix="/attachment", tags=["attachment-analysis"])
logger = logging.getLogger("zora.attachment")


def _resolve_sandbox_root() -> Path:
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir,
        base_dir.parent / "attachment-sandbox",
    ]
    for candidate in candidates:
        if (candidate / "app" / "static_analysis" / "pipeline.py").exists():
            return candidate
    return base_dir


_SANDBOX_ROOT = _resolve_sandbox_root()

DEFAULT_S3_REGION = os.getenv("AWS_REGION", "ap-south-1")
DEFAULT_S3_BUCKET = os.getenv("AWS_S3_BUCKET_NAME") or os.getenv("S3_BUCKET_NAME") or "zora-ai-dev-bucket"


def _build_s3_client():
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION", DEFAULT_S3_REGION)

    if not access_key or not secret_key:
        raise RuntimeError("AWS credentials are missing. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")

    return boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


def _upload_to_s3(local_path: str, original_filename: str) -> str:
    bucket = DEFAULT_S3_BUCKET
    if not bucket:
        raise RuntimeError("S3 bucket is not configured. Set AWS_S3_BUCKET_NAME or S3_BUCKET_NAME.")

    safe_name = os.path.basename(original_filename or "uploaded_attachment")
    object_key = f"attachment/{uuid.uuid4()}_{safe_name}"
    region = os.getenv("AWS_REGION", DEFAULT_S3_REGION)

    s3_client = _build_s3_client()
    s3_client.upload_file(local_path, bucket, object_key)

    return f"https://{bucket}.s3.{region}.amazonaws.com/{object_key}"


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
async def analyze_attachment(
    request: Request,
    file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
):
    if file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")

    filename = os.path.basename(file.filename or "uploaded_attachment")
    if not filename:
        filename = "uploaded_attachment"

    suffix = Path(filename).suffix
    temp_file_path: str | None = None
    attachment_request_row: AttachmentRequest | None = None

    print("[attachment-debug] /attachment/analyze request received")
    logger.info("Attachment analysis request received", extra={"upload_filename": filename})

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

        s3_url = _upload_to_s3(temp_file_path, filename)

        user_id_value = getattr(request.state, "user_id", None)
        user_id = None
        if user_id_value:
            try:
                user_id = uuid.UUID(str(user_id_value))
            except ValueError:
                logger.warning("Invalid user_id in request state, storing as null", extra={"user_id": user_id_value})

        attachment_request_row = AttachmentRequest(
            user_id=user_id,
            filename=filename,
            mime_type=file.content_type,
            file_size=len(content),
            s3_url=s3_url,
            status="uploaded",
        )
        db.add(attachment_request_row)
        db.commit()
        db.refresh(attachment_request_row)

        print(f"[attachment-debug] Request persisted request_id={attachment_request_row.id}")
        logger.info(
            "Attachment request persisted",
            extra={"request_id": str(attachment_request_row.id), "s3_url": s3_url},
        )

        run_static_pipeline = _load_pipeline_runner()
        report = run_static_pipeline(temp_file_path)

        if not isinstance(report, dict):
            raise RuntimeError("Attachment pipeline returned an invalid response")

        analysis_row = AttachmentAnalysis(
            request_id=attachment_request_row.id,
            final_verdict=str(report.get("final_verdict") or "unknown"),
            engines=json.dumps(report.get("engines") if isinstance(report.get("engines"), dict) else {}, default=str),
            features=json.dumps(report.get("features") if isinstance(report.get("features"), dict) else {}, default=str),
            status="completed",
            error_message=None,
        )
        attachment_request_row.status = "completed"
        db.add(analysis_row)
        db.commit()
        db.refresh(analysis_row)

        print(f"[attachment-debug] Analysis persisted analysis_id={analysis_row.id}")
        logger.info(
            "Attachment analysis completed and persisted",
            extra={"request_id": str(attachment_request_row.id), "analysis_id": str(analysis_row.id)},
        )

        return AttachmentAnalyzeResponse(
            request_id=attachment_request_row.id,
            analysis_id=analysis_row.id,
            filename=filename,
            file_size=len(content),
            s3_url=s3_url,
            status="completed",
            final_verdict=str(report.get("final_verdict") or "unknown"),
            engines=_normalize_engine_results(report.get("engines")),
            features=report.get("features") if isinstance(report.get("features"), dict) else {},
        )
    except HTTPException:
        raise
    except RuntimeError as exc:
        if attachment_request_row is not None:
            analysis_row = AttachmentAnalysis(
                request_id=attachment_request_row.id,
                final_verdict="unknown",
                engines=json.dumps({}, default=str),
                features=json.dumps({}, default=str),
                status="failed",
                error_message=str(exc),
            )
            attachment_request_row.status = "failed"
            db.add(analysis_row)
            db.commit()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        if attachment_request_row is not None:
            analysis_row = AttachmentAnalysis(
                request_id=attachment_request_row.id,
                final_verdict="unknown",
                engines=json.dumps({}, default=str),
                features=json.dumps({}, default=str),
                status="failed",
                error_message=str(exc),
            )
            attachment_request_row.status = "failed"
            db.add(analysis_row)
            db.commit()
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
