import torch
import librosa
import numpy as np
import io
import asyncio
import json
import logging
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.ai_security.guard import is_prompt_injection
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

# Custom modules
from app.database import get_db
from app.models import VoiceAnalysis, VoiceRequest
from app.schemas import VoiceAnalysisResponse
from app.voice_analysis.src.transcription import get_transcript
from app.voice_analysis.src.fraud_analyzer import analyze_fraud_intent
from app.voice_analysis.src.voice_model import ResNetBiLSTM

router = APIRouter(prefix="/voice", tags=["voice-analysis"])
executor = ThreadPoolExecutor(max_workers=3)
logger = logging.getLogger("zora.voice_analysis")


def _safe_load_json(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _normalize_history_analysis_payload(req: VoiceRequest, analysis: VoiceAnalysis) -> dict:
    voice_result = _safe_load_json(analysis.voice_result)
    fraud_report = _safe_load_json(analysis.fraud_report)

    # Keep response shape compatible with the /voice/analyse payload expected by frontend.
    normalized_voice = {
        "result": voice_result.get("result") or "Unknown",
        "confidence": float(voice_result.get("confidence") or 0.0),
        "pred_label": int(voice_result.get("pred_label") or 0),
        "chunks_analyzed": int(voice_result.get("chunks_analyzed") or 0),
    }

    normalized_fraud = {
        "risk_score": float(fraud_report.get("risk_score") or 0.0),
        "is_fraud": bool(fraud_report.get("is_fraud")),
        "system_logic": str(fraud_report.get("system_logic") or fraud_report.get("error") or "No analysis available."),
        "red_flags": fraud_report.get("red_flags") if isinstance(fraud_report.get("red_flags"), list) else [],
    }

    return {
        "request_id": req.id,
        "analysis_id": analysis.id,
        "status": analysis.status,
        "filename": req.filename,
        "voice_analysis": normalized_voice,
        "transcript": req.transcript or "",
        "fraud_report": normalized_fraud,
    }

# --- CONFIG ---
SR = 16000
DURATION = 3
N_MFCC = 40
MAX_LEN = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.pth"

model = ResNetBiLSTM().to(device)
model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
model.eval()


def split_audio(audio, sr, chunk_duration=3):
    chunk_size = int(sr * chunk_duration)
    chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
    # We only take full 3-second chunks to avoid length bias
    return [c for c in chunks if len(c) == chunk_size]

def preprocess_chunk(chunk, sr):
    """Matches the 3-channel training preprocessing exactly"""
    # 1. Extract MFCC, Delta, and Delta-Delta
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # 2. Stack into [3, 40, 300]
    features = np.stack([mfcc, delta, delta2], axis=0)
    
    # 3. Standardize/Normalize
    features = (features - np.mean(features)) / (np.std(features) + 1e-6)
    
    # 4. Convert to Tensor [Batch=1, Channels=3, Freq=40, Time=300]
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    return tensor.to(device)

def run_voice_model_logic(audio_bytes):
    """Standardized logic for ResNet-BiLSTM inference"""
    print("Running ResNet-BiLSTM Voice Analysis")
    
    # Load audio (force 16k)
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=SR)
    
    # Peak normalization
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    chunks = split_audio(audio, sr)
    
    # If audio is shorter than 3s, pad it to 3s for a single prediction
    if not chunks:
        target_len = SR * DURATION
        padded_audio = np.pad(audio, (0, target_len - len(audio)))
        chunks = [padded_audio]

    predictions, confidences = [], []
    
    with torch.no_grad():
        for chunk in chunks:
            input_tensor = preprocess_chunk(chunk, sr)
            output = model(input_tensor)
            
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
            
            predictions.append(pred)
            confidences.append(probs[0][pred].item())

    # Majority Voting
    final_prediction = max(set(predictions), key=predictions.count)
    # Average confidence for the winning class
    relevant_confs = [confidences[i] for i in range(len(predictions)) if predictions[i] == final_prediction]
    final_confidence = np.mean(relevant_confs)
    
    label_map = {0: "Real (Bonafide)", 1: "Spoof/Deepfake"}

    return {
        "result": label_map[final_prediction],
        "confidence": round(float(final_confidence), 4),
        "pred_label": int(final_prediction),
        "chunks_analyzed": len(chunks)
    }

@router.get("/history", status_code=status.HTTP_200_OK)
def get_voice_history(request: Request, db: Session = Depends(get_db)):
    user_id_value = getattr(request.state, "user_id", None)
    user_id = None
    if user_id_value:
        try:
            user_id = uuid.UUID(str(user_id_value))
        except ValueError:
            pass

    query = (
        db.query(VoiceRequest, VoiceAnalysis)
        .outerjoin(VoiceAnalysis, VoiceAnalysis.request_id == VoiceRequest.id)
    )
    if user_id:
        query = query.filter(VoiceRequest.user_id == user_id)

    rows = query.order_by(VoiceRequest.created_at.desc()).limit(20).all()

    history = []
    for req, analysis in rows:
        voice_result_label = None
        confidence = None
        risk_score = None
        is_fraud = None
        if analysis:
            try:
                vr = json.loads(analysis.voice_result) if analysis.voice_result else {}
                voice_result_label = vr.get("result")
                confidence = vr.get("confidence")
            except Exception:
                pass
            try:
                fr = json.loads(analysis.fraud_report) if analysis.fraud_report else {}
                risk_score = fr.get("risk_score")
                is_fraud = fr.get("is_fraud")
            except Exception:
                pass

        history.append({
            "request_id": str(req.id),
            "filename": req.filename,
            "status": req.status,
            "created_at": req.created_at.isoformat() if req.created_at else None,
            "transcript": (req.transcript or "")[:120],
            "voice_result": voice_result_label,
            "confidence": confidence,
            "risk_score": risk_score,
            "is_fraud": is_fraud,
        })
    return history


@router.get("/history/{request_id}", response_model=VoiceAnalysisResponse, status_code=status.HTTP_200_OK)
def get_voice_history_detail(request_id: str, request: Request, db: Session = Depends(get_db)):
    user_id_value = getattr(request.state, "user_id", None)
    user_id = None
    if user_id_value:
        try:
            user_id = uuid.UUID(str(user_id_value))
        except ValueError:
            pass

    try:
        req_uuid = uuid.UUID(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id") from exc

    query = (
        db.query(VoiceRequest, VoiceAnalysis)
        .join(VoiceAnalysis, VoiceAnalysis.request_id == VoiceRequest.id)
        .filter(VoiceRequest.id == req_uuid)
    )
    if user_id:
        query = query.filter(VoiceRequest.user_id == user_id)

    row = query.first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice analysis not found")

    req, analysis = row
    return _normalize_history_analysis_payload(req, analysis)


@router.delete("/history/{request_id}", status_code=status.HTTP_200_OK)
def delete_voice_history_item(request_id: str, request: Request, db: Session = Depends(get_db)):
    user_id_value = getattr(request.state, "user_id", None)
    user_id = None
    if user_id_value:
        try:
            user_id = uuid.UUID(str(user_id_value))
        except ValueError:
            pass

    try:
        req_uuid = uuid.UUID(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id") from exc

    query = db.query(VoiceRequest).filter(VoiceRequest.id == req_uuid)
    if user_id:
        query = query.filter(VoiceRequest.user_id == user_id)

    request_row = query.first()
    if not request_row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice analysis not found")

    db.query(VoiceAnalysis).filter(VoiceAnalysis.request_id == req_uuid).delete(synchronize_session=False)
    db.query(VoiceRequest).filter(VoiceRequest.id == req_uuid).delete(synchronize_session=False)
    db.commit()
    return {"status": "deleted", "request_id": request_id}


@router.delete("/history", status_code=status.HTTP_200_OK)
def clear_voice_history(request: Request, db: Session = Depends(get_db)):
    user_id_value = getattr(request.state, "user_id", None)
    user_id = None
    if user_id_value:
        try:
            user_id = uuid.UUID(str(user_id_value))
        except ValueError:
            pass

    base_query = db.query(VoiceRequest.id)
    if user_id:
        base_query = base_query.filter(VoiceRequest.user_id == user_id)

    request_ids = [row[0] for row in base_query.all()]
    if not request_ids:
        return {"status": "cleared", "deleted": 0}

    db.query(VoiceAnalysis).filter(VoiceAnalysis.request_id.in_(request_ids)).delete(synchronize_session=False)
    deleted_count = db.query(VoiceRequest).filter(VoiceRequest.id.in_(request_ids)).delete(synchronize_session=False)
    db.commit()
    return {"status": "cleared", "deleted": int(deleted_count or 0)}


# Main Endpoint
@router.post("/analyse", response_model=VoiceAnalysisResponse, status_code=status.HTTP_200_OK)
async def detect_fraud(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    print("[voice-debug] /voice/analyse request received")
    logger.info("Voice analysis request received", extra={"upload_filename": file.filename})

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    loop = asyncio.get_event_loop()

    voice_task = loop.run_in_executor(executor, run_voice_model_logic, audio_bytes)
    transcript_task = loop.run_in_executor(executor, get_transcript, io.BytesIO(audio_bytes))

    voice_res, transcript_text = await asyncio.gather(voice_task, transcript_task)

    user_id_value = getattr(request.state, "user_id", None)
    user_id = None
    if user_id_value:
        try:
            user_id = uuid.UUID(str(user_id_value))
        except ValueError:
            logger.warning("Invalid user_id in request state, storing as null", extra={"user_id": user_id_value})

    voice_request = VoiceRequest(
        user_id=user_id,
        filename=file.filename or "uploaded_audio",
        mime_type=file.content_type,
        file_size=len(audio_bytes),
        transcript=transcript_text or "",
        status="transcribed",
    )
    db.add(voice_request)
    db.commit()
    db.refresh(voice_request)

    print(f"[voice-debug] Transcription persisted request_id={voice_request.id}")
    logger.info(
        "Voice transcription persisted",
        extra={"request_id": str(voice_request.id), "transcript_length": len(transcript_text or "")},
    )

    if "error" in voice_res:
        failure_detail = str(voice_res.get("error") or "Voice model inference failed")
        analysis_row = VoiceAnalysis(
            request_id=voice_request.id,
            voice_result=json.dumps(voice_res, default=str),
            fraud_report=json.dumps({"error": failure_detail}),
            status="failed",
            error_message=failure_detail,
        )
        voice_request.status = "failed"
        db.add(analysis_row)
        db.commit()

        logger.error("Voice model failed", extra={"request_id": str(voice_request.id), "detail": failure_detail})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=failure_detail)

    # Fraud Analysis via LLM
    try:
        llm_output_raw = analyze_fraud_intent(
            transcript_text,
            voice_res["result"],
            voice_res["confidence"],
        )

        try:
            fraud_report = json.loads(llm_output_raw)
        except Exception:  # noqa: BLE001
            fraud_report = {"error": "JSON parsing failed", "raw": llm_output_raw}
    except Exception as exc:  # noqa: BLE001
        failure_detail = f"Fraud intent analysis failed: {exc}"
        analysis_row = VoiceAnalysis(
            request_id=voice_request.id,
            voice_result=json.dumps(voice_res, default=str),
            fraud_report=json.dumps({"error": failure_detail}),
            status="failed",
            error_message=failure_detail,
        )
        voice_request.status = "failed"
        db.add(analysis_row)
        db.commit()

        logger.error(
            "Voice fraud analysis failed",
            extra={"request_id": str(voice_request.id), "detail": failure_detail},
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=failure_detail) from exc

    analysis_row = VoiceAnalysis(
        request_id=voice_request.id,
        voice_result=json.dumps(voice_res, default=str),
        fraud_report=json.dumps(fraud_report, default=str),
        status="completed",
        error_message=None,
    )
    voice_request.status = "completed"
    db.add(analysis_row)
    db.commit()
    db.refresh(analysis_row)

    print(f"[voice-debug] Analysis persisted analysis_id={analysis_row.id}")
    logger.info(
        "Voice analysis completed and persisted",
        extra={"request_id": str(voice_request.id), "analysis_id": str(analysis_row.id)},
    )

    return VoiceAnalysisResponse(
        request_id=voice_request.id,
        analysis_id=analysis_row.id,
        status=analysis_row.status,
        filename=file.filename or "uploaded_audio",
        voice_analysis=voice_res,
        transcript=transcript_text,
        fraud_report=fraud_report,
    )
