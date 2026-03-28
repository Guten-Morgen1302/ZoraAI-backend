from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.models import AttachmentAnalysis, AttachmentRequest
from app.schemas import AttachmentAnalyzeResponse, AttachmentEngineResult
from app.attachment_sandbox.llm_reasoner import explain_attachment_with_llm

router = APIRouter(prefix="/attachment", tags=["attachment-analysis"])

_SANDBOX_ROOT = Path(__file__).resolve().parent
_LLM_METADATA_KEY = "__llm_metadata"

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


def _safe_json_loads(raw: str | None) -> dict[str, object]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _safe_user_uuid(request: Request) -> uuid.UUID | None:
    state_user = request.state.user_id if hasattr(request.state, "user_id") else None
    if state_user is None:
        return None
    if isinstance(state_user, uuid.UUID):
        return state_user

    try:
        return uuid.UUID(str(state_user))
    except (TypeError, ValueError):
        return None


def _count_flagged_engines(engines: dict[str, object]) -> int:
    flagged = 0
    for payload in engines.values():
        if isinstance(payload, dict) and bool(payload.get("is_flagged", False)):
            flagged += 1
    return flagged


def _safe_rollback(db: Session) -> None:
    try:
        db.rollback()
    except Exception:  # noqa: BLE001
        pass


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


def _run_attachment_analysis_job(
    request_id: uuid.UUID,
    temp_file_path: str,
    filename: str,
    file_size: int,
    with_llm_explanation: bool,
) -> None:
    db = SessionLocal()
    try:
        request_row = db.query(AttachmentRequest).filter(AttachmentRequest.id == request_id).first()
        if request_row is None:
            return

        run_static_pipeline = _load_pipeline_runner()

        def _update_status(next_status: str) -> None:
            request_row.status = next_status
            db.add(request_row)
            db.commit()

        report = run_static_pipeline(temp_file_path, progress_callback=_update_status)

        if not isinstance(report, dict):
            raise RuntimeError("Attachment pipeline returned an invalid response")

        response_kwargs = {
            "filename": filename,
            "file_size": file_size,
            "final_verdict": str(report.get("final_verdict") or "unknown"),
            "engines": _normalize_engine_results(report.get("engines")),
            "features": report.get("features") if isinstance(report.get("features"), dict) else {},
        }

        if with_llm_explanation:
            _update_status("processing_llm")
            llm_result = explain_attachment_with_llm(report, filename)
            response_kwargs.update(
                {
                    "llm_enhanced": True,
                    "llm_label": llm_result.get("final_label"),
                    "llm_confidence": llm_result.get("confidence"),
                    "llm_explanation": llm_result.get("explanation"),
                    "llm_key_indicators": llm_result.get("key_indicators", []),
                    "llm_recommendations": llm_result.get("recommendations", []),
                }
            )

        features_payload = dict(response_kwargs["features"])
        if response_kwargs.get("llm_enhanced"):
            features_payload[_LLM_METADATA_KEY] = {
                "llm_enhanced": bool(response_kwargs.get("llm_enhanced", False)),
                "llm_label": response_kwargs.get("llm_label"),
                "llm_confidence": response_kwargs.get("llm_confidence"),
                "llm_explanation": response_kwargs.get("llm_explanation"),
                "llm_key_indicators": response_kwargs.get("llm_key_indicators", []),
                "llm_recommendations": response_kwargs.get("llm_recommendations", []),
            }

        engines_payload = {
            name: {
                "is_flagged": value.is_flagged,
                "hits": value.hits,
                "signature": value.signature,
                "score": value.score,
            }
            for name, value in response_kwargs["engines"].items()
        }

        analysis_row = AttachmentAnalysis(
            request_id=request_row.id,
            final_verdict=response_kwargs["final_verdict"],
            engines=json.dumps(engines_payload, default=str),
            features=json.dumps(features_payload, default=str),
            status="completed",
            error_message=None,
        )

        request_row.status = "completed"
        db.add(request_row)
        db.add(analysis_row)
        db.commit()
    except Exception as exc:  # noqa: BLE001
        _safe_rollback(db)

        request_row = db.query(AttachmentRequest).filter(AttachmentRequest.id == request_id).first()
        if request_row is not None:
            request_row.status = "failed"
            db.add(request_row)
            db.commit()

            existing_analysis = db.query(AttachmentAnalysis).filter(AttachmentAnalysis.request_id == request_row.id).first()
            if existing_analysis is None:
                failed_analysis = AttachmentAnalysis(
                    request_id=request_row.id,
                    final_verdict="unknown",
                    engines=json.dumps({}, default=str),
                    features=json.dumps({}, default=str),
                    status="failed",
                    error_message=str(exc),
                )
                db.add(failed_analysis)
                db.commit()
    finally:
        db.close()
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


@router.post("/analyze", response_model=AttachmentAnalyzeResponse, status_code=status.HTTP_200_OK)
async def analyze_attachment(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    file: UploadFile | None = File(default=None),
    with_llm_explanation: str = Form("false"),
):
    if file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")

    filename = os.path.basename(file.filename or "uploaded_attachment")
    if not filename:
        filename = "uploaded_attachment"

    mime_type = file.content_type
    suffix = Path(filename).suffix
    temp_file_path: str | None = None
    request_row: AttachmentRequest | None = None
    file_size = 0
    is_llm_requested = with_llm_explanation.strip().lower() in ("true", "1", "yes", "y", "on")

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
            file_size = len(content)

        request_row = AttachmentRequest(
            user_id=_safe_user_uuid(request),
            filename=filename,
            mime_type=mime_type,
            file_size=file_size,
            # Keep compatibility with existing DBs that still enforce NOT NULL.
            s3_url="",
            status="queued",
        )
        db.add(request_row)
        db.commit()
        db.refresh(request_row)

        background_tasks.add_task(
            _run_attachment_analysis_job,
            request_row.id,
            temp_file_path,
            filename,
            file_size,
            is_llm_requested,
        )
        # Background task is now responsible for temp file cleanup.
        temp_file_path = None

        return AttachmentAnalyzeResponse(
            request_id=request_row.id,
            analysis_id=None,
            filename=filename,
            file_size=file_size,
            s3_url=request_row.s3_url,
            status=request_row.status,
            final_verdict="processing",
            engines={},
            features={},
            llm_enhanced=is_llm_requested,
        )
    except HTTPException:
        _safe_rollback(db)
        if request_row is not None:
            request_row.status = "failed"
            db.add(request_row)
            db.commit()
        raise
    except RuntimeError as exc:
        _safe_rollback(db)
        if request_row is not None:
            request_row.status = "failed"
            db.add(request_row)
            db.commit()

            failed_analysis = AttachmentAnalysis(
                request_id=request_row.id,
                final_verdict="unknown",
                engines=json.dumps({}, default=str),
                features=json.dumps({}, default=str),
                status="failed",
                error_message=str(exc),
            )
            db.add(failed_analysis)
            db.commit()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _safe_rollback(db)
        if request_row is not None:
            request_row.status = "failed"
            db.add(request_row)
            db.commit()

            failed_analysis = AttachmentAnalysis(
                request_id=request_row.id,
                final_verdict="unknown",
                engines=json.dumps({}, default=str),
                features=json.dumps({}, default=str),
                status="failed",
                error_message=str(exc),
            )
            db.add(failed_analysis)
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


@router.get("/history", status_code=status.HTTP_200_OK)
def get_attachment_history(request: Request, db: Session = Depends(get_db)):
    user_id = _safe_user_uuid(request)

    query = (
        db.query(AttachmentRequest, AttachmentAnalysis)
        .outerjoin(AttachmentAnalysis, AttachmentAnalysis.request_id == AttachmentRequest.id)
    )
    if user_id is not None:
        query = query.filter(AttachmentRequest.user_id == user_id)

    rows = query.order_by(AttachmentRequest.created_at.desc()).limit(20).all()

    history: list[dict[str, object]] = []
    for request_row, analysis_row in rows:
        engines = _safe_json_loads(analysis_row.engines) if analysis_row else {}
        clamav_signature = None
        clam_payload = engines.get("clamav") if isinstance(engines, dict) else None
        if isinstance(clam_payload, dict):
            raw_signature = clam_payload.get("signature")
            if isinstance(raw_signature, str) and raw_signature.strip():
                clamav_signature = raw_signature.strip()

        history.append(
            {
                "request_id": str(request_row.id),
                "filename": request_row.filename,
                "file_size": request_row.file_size,
                "created_at": request_row.created_at.isoformat() if request_row.created_at else None,
                "status": request_row.status,
                "final_verdict": analysis_row.final_verdict if analysis_row else None,
                "flagged_engines": _count_flagged_engines(engines),
                "clamav_signature": clamav_signature,
            }
        )

    return history


@router.get("/history/{request_id}", response_model=AttachmentAnalyzeResponse, status_code=status.HTTP_200_OK)
def get_attachment_history_detail(request_id: str, request: Request, db: Session = Depends(get_db)):
    user_id = _safe_user_uuid(request)

    try:
        request_uuid = uuid.UUID(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id") from exc

    query = (
        db.query(AttachmentRequest, AttachmentAnalysis)
        .outerjoin(AttachmentAnalysis, AttachmentAnalysis.request_id == AttachmentRequest.id)
        .filter(AttachmentRequest.id == request_uuid)
    )
    if user_id is not None:
        query = query.filter(AttachmentRequest.user_id == user_id)

    row = query.first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attachment analysis not found")

    request_row, analysis_row = row
    engines = _safe_json_loads(analysis_row.engines) if analysis_row else {}
    features = _safe_json_loads(analysis_row.features) if analysis_row else {}
    llm_metadata: dict[str, object] = {}
    if isinstance(features, dict):
        raw_llm_metadata = features.pop(_LLM_METADATA_KEY, None)
        if isinstance(raw_llm_metadata, dict):
            llm_metadata = raw_llm_metadata

    payload = {
        "request_id": request_row.id,
        "analysis_id": analysis_row.id if analysis_row else None,
        "filename": request_row.filename,
        "file_size": request_row.file_size,
        "s3_url": request_row.s3_url,
        "status": request_row.status,
        "final_verdict": analysis_row.final_verdict if analysis_row else "processing",
        "engines": _normalize_engine_results(engines),
        "features": features if isinstance(features, dict) else {},
        "llm_enhanced": bool(llm_metadata.get("llm_enhanced", False)),
        "llm_label": llm_metadata.get("llm_label"),
        "llm_confidence": llm_metadata.get("llm_confidence"),
        "llm_explanation": llm_metadata.get("llm_explanation"),
        "llm_key_indicators": llm_metadata.get("llm_key_indicators") if isinstance(llm_metadata.get("llm_key_indicators"), list) else [],
        "llm_recommendations": llm_metadata.get("llm_recommendations") if isinstance(llm_metadata.get("llm_recommendations"), list) else [],
    }

    return AttachmentAnalyzeResponse(**payload)


@router.delete("/history/{request_id}", status_code=status.HTTP_200_OK)
def delete_attachment_history_item(request_id: str, request: Request, db: Session = Depends(get_db)):
    user_id = _safe_user_uuid(request)

    try:
        request_uuid = uuid.UUID(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id") from exc

    query = db.query(AttachmentRequest).filter(AttachmentRequest.id == request_uuid)
    if user_id is not None:
        query = query.filter(AttachmentRequest.user_id == user_id)

    request_row = query.first()
    if not request_row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attachment analysis not found")

    db.query(AttachmentAnalysis).filter(AttachmentAnalysis.request_id == request_uuid).delete(synchronize_session=False)
    db.query(AttachmentRequest).filter(AttachmentRequest.id == request_uuid).delete(synchronize_session=False)
    db.commit()
    return {"status": "deleted", "request_id": request_id}


@router.delete("/history", status_code=status.HTTP_200_OK)
def clear_attachment_history(request: Request, db: Session = Depends(get_db)):
    user_id = _safe_user_uuid(request)

    base_query = db.query(AttachmentRequest.id)
    if user_id is not None:
        base_query = base_query.filter(AttachmentRequest.user_id == user_id)

    request_ids = [row[0] for row in base_query.all()]
    if not request_ids:
        return {"status": "cleared", "deleted": 0}

    db.query(AttachmentAnalysis).filter(AttachmentAnalysis.request_id.in_(request_ids)).delete(synchronize_session=False)
    deleted_count = db.query(AttachmentRequest).filter(AttachmentRequest.id.in_(request_ids)).delete(synchronize_session=False)
    db.commit()
    return {"status": "cleared", "deleted": int(deleted_count or 0)}
