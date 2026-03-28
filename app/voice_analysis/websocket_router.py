import asyncio
import io
import json
import time
import numpy as np
import librosa
import soundfile as sf
import torch
import tempfile
import os
from pathlib import Path
from pydub import AudioSegment
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from concurrent.futures import ThreadPoolExecutor

from app.voice_analysis.src.fraud_analyzer import analyze_fraud_intent
from app.voice_analysis.src.voice_model import ResNetBiLSTM

ws_router = APIRouter(prefix="/voice/ws", tags=["voice-analysis-ws"])
executor = ThreadPoolExecutor(max_workers=6)

print("[INIT] websocket_router module loaded")

# --- CONFIG ---
SR = 16000
CHUNK_DURATION = 3
N_MFCC = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.pth"
voice_model = ResNetBiLSTM().to(device)
voice_model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
voice_model.eval()


# ──────────────────────────────────────────
#  HELPER FUNCTIONS
# ──────────────────────────────────────────

def decode_webm_to_numpy(byte_data: bytes) -> np.ndarray:
    webm_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
            f.write(byte_data)
            webm_path = f.name
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio = audio.set_frame_rate(SR).set_channels(1).set_sample_width(2)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        return samples
    except Exception as e:
        print(f"[DECODE] ❌ ffmpeg failed — {e}")
        raise
    finally:
        if webm_path and os.path.exists(webm_path):
            os.unlink(webm_path)


def preprocess_chunk(chunk, sr):
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.stack([mfcc, delta, delta2], axis=0)
    features = (features - np.mean(features)) / (np.std(features) + 1e-6)
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)


def numpy_to_wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format='WAV', subtype='PCM_16')
    buf.seek(0)
    return buf.read()


def _transcribe_from_bytes(wav_bytes: bytes) -> str:
    from app.voice_analysis.src import transcription as _t
    buf = io.BytesIO(wav_bytes)
    segments, info = _t.model.transcribe(buf, beam_size=1)
    text = " ".join([seg.text for seg in segments])
    return text.strip()


def run_voice_model_logic_on_numpy(chunk):
    target_samples = SR * CHUNK_DURATION
    if len(chunk) < target_samples:
        chunk = np.pad(chunk, (0, target_samples - len(chunk)))
    elif len(chunk) > target_samples:
        chunk = chunk[:target_samples]
    if np.max(np.abs(chunk)) > 0:
        chunk = chunk / np.max(np.abs(chunk))
    with torch.no_grad():
        input_tensor = preprocess_chunk(chunk, SR)
        output = voice_model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).item()
    return {
        "result": "Spoof/Deepfake" if pred == 1 else "Real",
        "confidence": float(probs[0][pred].item()),
        "pred_label": int(pred)
    }


# ──────────────────────────────────────────
#  SAFE WS SEND
# ──────────────────────────────────────────

async def safe_send_json(websocket: WebSocket, payload: dict) -> bool:
    """Send JSON to WS. Returns True if sent, False if dead."""
    try:
        await websocket.send_json(payload)
        return True
    except Exception:
        return False


# ──────────────────────────────────────────
#  PROCESSING PIPELINE
# ──────────────────────────────────────────

async def process_numpy_and_send(
    websocket: WebSocket,
    samples: np.ndarray,
    loop: asyncio.AbstractEventLoop,
    dispatch_id: int,
    transcript_accumulator: list,
    pending_tasks: list,
):
    """Process audio: [voice_model ∥ whisper] → [send_to_frontend ∥ accumulate_for_llm]

    Layer 3 (fast send) and Layer 4 (LLM) now run in PARALLEL.
    LLM is batched — only fires when enough transcript has accumulated.
    """
    t0 = time.time()
    print(f"\n[LAYER 2: MODEL+WHISPER] dispatch #{dispatch_id} | {len(samples)/SR:.1f}s | starting...")

    try:
        if len(samples) > SR * CHUNK_DURATION:
            samples = samples[:SR * CHUNK_DURATION]

        # --- PARALLEL: voice model + Whisper ---
        async def run_voice():
            return await loop.run_in_executor(executor, run_voice_model_logic_on_numpy, samples)

        async def run_whisper():
            wav_bytes = await loop.run_in_executor(executor, numpy_to_wav_bytes, samples, SR)
            return await loop.run_in_executor(executor, _transcribe_from_bytes, wav_bytes)

        voice_res, transcript = await asyncio.gather(run_voice(), run_whisper())
        t1 = time.time()
        print(f"[LAYER 2: MODEL+WHISPER] dispatch #{dispatch_id} | ✅ {t1-t0:.1f}s | voice={voice_res['result']}({voice_res['confidence']:.2f}) | '{transcript[:60]}'")

        # --- Build fast payload ---
        fast_payload = {
            "transcript": str(transcript or ""),
            "voice_label": str(voice_res['result']),
            "confidence": float(voice_res['confidence']),
            "risk_score": 0,
            "warning": "",
            "is_spoof": bool(voice_res['pred_label'] == 1)
        }

        # --- PARALLEL: send fast payload to frontend + accumulate for LLM ---
        # These two are independent — send the transcript instantly while
        # building up the LLM batch in the background.

        async def layer3_fast_send():
            sent = await safe_send_json(websocket, fast_payload)
            if sent:
                print(f"[LAYER 3: FAST SEND] dispatch #{dispatch_id} | ⚡ sent in {time.time()-t0:.1f}s")
            else:
                print(f"[LAYER 3: FAST SEND] dispatch #{dispatch_id} | WS closed, dropped")

        async def layer4_llm_batch():
            if transcript and transcript.strip():
                transcript_accumulator.append({
                    "text": transcript,
                    "voice_result": voice_res['result'],
                    "confidence": voice_res['confidence'],
                    "dispatch_id": dispatch_id,
                })

            # Fire LLM when ≥3 segments OR ≥15 words accumulated
            total_words = sum(len(t["text"].split()) for t in transcript_accumulator)
            if len(transcript_accumulator) >= 3 or total_words >= 15:
                combined_text = " ".join(t["text"] for t in transcript_accumulator)
                latest = transcript_accumulator[-1]
                batch_ids = [t["dispatch_id"] for t in transcript_accumulator]
                transcript_accumulator.clear()

                print(f"[LAYER 4: LLM] dispatches {batch_ids} | {len(combined_text.split())} words → Gemini")
                t2 = time.time()
                llm_raw = await loop.run_in_executor(
                    executor, analyze_fraud_intent, combined_text, latest["voice_result"], latest["confidence"]
                )
                fraud_report = {"risk_score": 0, "system_logic": ""}
                try:
                    fraud_report = json.loads(llm_raw)
                except json.JSONDecodeError:
                    pass

                full_payload = {
                    "transcript": combined_text,
                    "voice_label": str(latest["voice_result"]),
                    "confidence": float(latest["confidence"]),
                    "risk_score": int(fraud_report.get("risk_score", 0)),
                    "warning": str(fraud_report.get("system_logic", "")),
                    "is_spoof": bool(latest["voice_result"] == "Spoof/Deepfake")
                }
                sent = await safe_send_json(websocket, full_payload)
                if sent:
                    print(f"[LAYER 4: LLM SEND] dispatches {batch_ids} | ✅ risk={full_payload['risk_score']} in {time.time()-t2:.1f}s")
                else:
                    print(f"[LAYER 4: LLM SEND] dispatches {batch_ids} | WS closed, dropped")

        # Run Layer 3 + Layer 4 in PARALLEL
        await asyncio.gather(layer3_fast_send(), layer4_llm_batch())

    except Exception as e:
        print(f"[ERROR] dispatch #{dispatch_id} | {e}")
    finally:
        # Remove self from pending tasks
        current = asyncio.current_task()
        if current in pending_tasks:
            pending_tasks.remove(current)


# ──────────────────────────────────────────
#  WEBSOCKET ENDPOINT
# ──────────────────────────────────────────

@ws_router.websocket("/live-protect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[CONNECTION] ✅ WebSocket accepted")
    chunk_count = 0
    dispatch_count = 0

    # Shared state
    transcript_accumulator = []
    pending_tasks = []   # track background tasks so we can await them on stop

    all_bytes = b""
    total_samples_processed = 0
    numpy_buffer = np.array([], dtype=np.float32)
    SAMPLES_THRESHOLD = SR * 3
    SAMPLES_KEEP_TAIL = SR * 1

    loop = asyncio.get_running_loop()
    capture_stopped = False

    try:
        while True:
            try:
                raw = await websocket.receive()
            except WebSocketDisconnect:
                print("[CONNECTION] 🔌 Client disconnected")
                break

            if raw.get("type") == "websocket.receive" and "text" in raw:
                text_msg = raw["text"]
                if text_msg == "STOP_CAPTURE":
                    print("[CONNECTION] 🛑 STOP_CAPTURE received — finishing pending work...")
                    capture_stopped = True
                    break
                continue

            if "bytes" not in raw or raw["bytes"] is None:
                continue

            data = raw["bytes"]
            chunk_count += 1
            all_bytes += data
            print(f"[LAYER 1: CAPTURE] chunk #{chunk_count} | {len(data)} bytes | total={len(all_bytes)} bytes")

            try:
                # Decode the entire stream so far to ensure valid EBML/WebM structure
                current_full_audio = await loop.run_in_executor(executor, decode_webm_to_numpy, all_bytes)
                
                # Extract only the NEW samples
                if len(current_full_audio) > total_samples_processed:
                    new_samples = current_full_audio[total_samples_processed:]
                    total_samples_processed = len(current_full_audio)
                    numpy_buffer = np.concatenate([numpy_buffer, new_samples])
                    print(f"[LAYER 1: BUFFER] chunk #{chunk_count} | new={len(new_samples)/SR:.1f}s | buffer={len(numpy_buffer)/SR:.1f}s")
                else:
                    print(f"[LAYER 1: BUFFER] chunk #{chunk_count} | no new samples decoded")

            except Exception as decode_err:
                # Often the last few bytes of a chunk are incomplete; we wait for next chunk
                print(f"[LAYER 1: DECODE] chunk #{chunk_count} | ⚠️ {decode_err} (waiting for more data)")
                continue

            # --- DISPATCH ---
            if len(numpy_buffer) >= SAMPLES_THRESHOLD:
                dispatch_samples = numpy_buffer.copy()
                # Keep 1s overlap for smoother analysis if needed, or just clear
                numpy_buffer = numpy_buffer[-SAMPLES_KEEP_TAIL:]
                dispatch_count += 1
                print(f"[DISPATCH] 🔥 dispatch #{dispatch_count} | {len(dispatch_samples)/SR:.1f}s → Layer 2")
                task = asyncio.create_task(process_numpy_and_send(
                    websocket, dispatch_samples, loop, dispatch_count,
                    transcript_accumulator, pending_tasks
                ))
                pending_tasks.append(task)

    except Exception as e:
        print(f"[ERROR] WebSocket loop — {e}")

    # --- POST-CAPTURE: process any remaining buffer ---
    if len(numpy_buffer) >= SR * 1:  # process if ≥1s of audio left
        dispatch_count += 1
        print(f"[DISPATCH] 🔥 dispatch #{dispatch_count} (final) | {len(numpy_buffer)/SR:.1f}s → Layer 2")
        task = asyncio.create_task(process_numpy_and_send(
            websocket, numpy_buffer.copy(), loop, dispatch_count,
            transcript_accumulator, pending_tasks
        ))
        pending_tasks.append(task)

    # --- Wait for ALL pending tasks to finish ---
    if pending_tasks:
        print(f"[CONNECTION] ⏳ Waiting for {len(pending_tasks)} pending tasks to finish...")
        await asyncio.gather(*pending_tasks, return_exceptions=True)
        print(f"[CONNECTION] ✅ All tasks finished")

    # --- Flush any remaining transcripts to LLM ---
    if transcript_accumulator:
        combined_text = " ".join(t["text"] for t in transcript_accumulator)
        latest = transcript_accumulator[-1]
        batch_ids = [t["dispatch_id"] for t in transcript_accumulator]
        transcript_accumulator.clear()
        print(f"[FLUSH LLM] dispatches {batch_ids} | {len(combined_text.split())} words → Gemini")
        try:
            llm_raw = await loop.run_in_executor(
                executor, analyze_fraud_intent, combined_text, latest["voice_result"], latest["confidence"]
            )
            fraud_report = {"risk_score": 0, "system_logic": ""}
            try:
                fraud_report = json.loads(llm_raw)
            except json.JSONDecodeError:
                pass
            full_payload = {
                "transcript": combined_text,
                "voice_label": str(latest["voice_result"]),
                "confidence": float(latest["confidence"]),
                "risk_score": int(fraud_report.get("risk_score", 0)),
                "warning": str(fraud_report.get("system_logic", "")),
                "is_spoof": bool(latest["voice_result"] == "Spoof/Deepfake")
            }
            sent = await safe_send_json(websocket, full_payload)
            if sent:
                print(f"[FLUSH LLM] ✅ risk={full_payload['risk_score']} sent")
            else:
                print(f"[FLUSH LLM] WS closed, dropped")
        except Exception as e:
            print(f"[FLUSH LLM] ❌ {e}")

    # --- Close WS from server side ---
    try:
        await websocket.close()
    except Exception:
        pass

    print(f"[CONNECTION] 🔚 Done | chunks={chunk_count} dispatches={dispatch_count}")