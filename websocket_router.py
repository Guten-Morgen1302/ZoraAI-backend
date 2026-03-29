import asyncio
import io
import json
import time
import numpy as np
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
from app.voice_analysis.src.voice_model import WavLMClassifier

ws_router = APIRouter(prefix="/voice/ws", tags=["voice-analysis-ws"])
executor = ThreadPoolExecutor(max_workers=6)

print("[INIT] websocket_router module loaded")

# --- CONFIG ---
SR = 16000
CHUNK_DURATION = 3
MODEL_CHUNK_DURATION = 4
FRAUD_BATCH_MIN_SEGMENTS = 4
FRAUD_BATCH_MIN_WORDS = 24
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path(__file__).resolve().parent / "models" / "wavlm_asv2025_pruned.pth"
voice_model = WavLMClassifier().to(device)
if MODEL_PATH.exists():
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


def split_audio(samples: np.ndarray, sr: int, chunk_duration: int = MODEL_CHUNK_DURATION) -> list[np.ndarray]:
    chunk_size = int(sr * chunk_duration)
    chunks = [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]
    return [chunk for chunk in chunks if len(chunk) == chunk_size]


def preprocess_waveform(chunk: np.ndarray) -> torch.Tensor:
    if np.max(np.abs(chunk)) > 0:
        chunk = chunk / np.max(np.abs(chunk))
    return torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)


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
    target_samples = SR * MODEL_CHUNK_DURATION
    chunks = split_audio(chunk, SR, chunk_duration=MODEL_CHUNK_DURATION)
    if not chunks:
        padded = np.pad(chunk, (0, max(0, target_samples - len(chunk))))
        chunks = [padded[:target_samples]]

    predictions = []
    confidences = []

    with torch.no_grad():
        for model_chunk in chunks:
            input_tensor = preprocess_waveform(model_chunk)
            output = voice_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
            predictions.append(pred)
            confidences.append(probs[0][pred].item())

    final_prediction = max(set(predictions), key=predictions.count)
    relevant_confs = [
        confidences[i]
        for i in range(len(predictions))
        if predictions[i] == final_prediction
    ]
    final_confidence = float(np.mean(relevant_confs)) if relevant_confs else 0.0

    label_map = {0: "Real (Bonafide)", 1: "Spoof/Deepfake"}

    return {
        "result": label_map.get(final_prediction, "Unknown"),
        "confidence": final_confidence,
        "pred_label": int(final_prediction),
        "chunks_analyzed": len(chunks),
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
    full_transcript_segments: list,
    voice_state: dict,
    risk_state: dict,
    state_lock: asyncio.Lock,
    pending_tasks: list,
):
    """Process audio: [voice_model ∥ whisper] → [send_to_frontend ∥ accumulate_for_llm]

    Layer 3 (fast send) and Layer 4 (LLM) now run in PARALLEL.
    LLM is batched — only fires when enough transcript has accumulated.
    """
    t0 = time.time()
    print(f"\n[LAYER 2: MODEL+WHISPER] dispatch #{dispatch_id} | {len(samples)/SR:.1f}s | starting...")

    try:
        if len(samples) > SR * MODEL_CHUNK_DURATION:
            samples = samples[:SR * MODEL_CHUNK_DURATION]

        # --- PARALLEL: voice model + Whisper ---
        async def run_voice():
            return await loop.run_in_executor(executor, run_voice_model_logic_on_numpy, samples)

        async def run_whisper():
            wav_bytes = await loop.run_in_executor(executor, numpy_to_wav_bytes, samples, SR)
            return await loop.run_in_executor(executor, _transcribe_from_bytes, wav_bytes)

        voice_res, transcript = await asyncio.gather(run_voice(), run_whisper())
        t1 = time.time()
        print(f"[LAYER 2: MODEL+WHISPER] dispatch #{dispatch_id} | ✅ {t1-t0:.1f}s | voice={voice_res['result']}({voice_res['confidence']:.2f}) | '{transcript[:60]}'")

        async with state_lock:
            voice_state["voice_result"] = str(voice_res["result"])
            voice_state["confidence"] = float(voice_res["confidence"])

        async with state_lock:
            sticky_risk = int(risk_state.get("max_risk_score", 0))
            sticky_warning = str(risk_state.get("warning", ""))

        # --- Build fast payload ---
        fast_payload = {
            "transcript": str(transcript or ""),
            "voice_label": str(voice_res['result']),
            "confidence": float(voice_res['confidence']),
            "risk_score": sticky_risk,
            "warning": sticky_warning,
            "is_spoof": bool(voice_res['pred_label'] == 1),
            "final_settlement": False,
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
            llm_batch = []
            if transcript and transcript.strip():
                async with state_lock:
                    transcript_accumulator.append({
                        "text": transcript,
                        "voice_result": voice_res['result'],
                        "confidence": voice_res['confidence'],
                        "dispatch_id": dispatch_id,
                    })
                    full_transcript_segments.append(transcript)

            # Fire LLM on larger transcript batches to avoid over-reacting to tiny chunks.
            async with state_lock:
                total_words = sum(len(t["text"].split()) for t in transcript_accumulator)
                if len(transcript_accumulator) >= FRAUD_BATCH_MIN_SEGMENTS or total_words >= FRAUD_BATCH_MIN_WORDS:
                    llm_batch = transcript_accumulator.copy()
                    transcript_accumulator.clear()

            if llm_batch:
                combined_text = " ".join(t["text"] for t in llm_batch)
                latest = llm_batch[-1]
                batch_ids = [t["dispatch_id"] for t in llm_batch]

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

                raw_risk_score = int(fraud_report.get("risk_score", 0))
                raw_warning = str(fraud_report.get("system_logic", ""))
                async with state_lock:
                    if raw_risk_score > int(risk_state.get("max_risk_score", 0)):
                        risk_state["max_risk_score"] = raw_risk_score
                        if raw_warning:
                            risk_state["warning"] = raw_warning
                    sticky_risk_score = int(risk_state.get("max_risk_score", 0))
                    sticky_warning_text = str(risk_state.get("warning", "") or raw_warning)

                full_payload = {
                    "transcript": combined_text,
                    "voice_label": str(latest["voice_result"]),
                    "confidence": float(latest["confidence"]),
                    "risk_score": sticky_risk_score,
                    "warning": sticky_warning_text,
                    "is_spoof": bool(latest["voice_result"] == "Spoof/Deepfake"),
                    "final_settlement": False,
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
    full_transcript_segments = []
    voice_state = {"voice_result": "Unknown", "confidence": 0.0}
    risk_state = {"max_risk_score": 0, "warning": ""}
    state_lock = asyncio.Lock()
    pending_tasks = []   # track background tasks so we can await them on stop

    init_segment: bytes | None = None
    numpy_buffer = np.array([], dtype=np.float32)
    SAMPLES_THRESHOLD = SR * 3
    SAMPLES_KEEP_TAIL = SR * 1

    loop = asyncio.get_running_loop()
    capture_stopped = False

    try:
        while True:
            try:
                # receive_bytes() blocks until data OR disconnect
                raw = await websocket.receive()
            except WebSocketDisconnect:
                print("[CONNECTION] 🔌 Client disconnected")
                break

            # Check if frontend sent "STOP_CAPTURE" text message
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
            print(f"[LAYER 1: CAPTURE] chunk #{chunk_count} | {len(data)} bytes")

            if chunk_count == 1:
                init_segment = data
                print(f"[LAYER 1: CAPTURE] chunk #1 | saved init_segment ({len(init_segment)} bytes)")
                decodable = data
            else:
                if init_segment is None:
                    continue
                decodable = init_segment + data

            try:
                new_samples = await loop.run_in_executor(executor, decode_webm_to_numpy, decodable)
                if chunk_count > 1 and len(new_samples) > 0:
                    init_audio_len = int(SR * 0.5)
                    new_samples = new_samples[min(init_audio_len, len(new_samples) - 1):]
                numpy_buffer = np.concatenate([numpy_buffer, new_samples])
                print(f"[LAYER 1: BUFFER] chunk #{chunk_count} | buffer={len(numpy_buffer)/SR:.1f}s / need {SAMPLES_THRESHOLD/SR:.0f}s")
            except Exception as decode_err:
                print(f"[LAYER 1: DECODE] chunk #{chunk_count} | ❌ {decode_err}")
                continue

            # --- DISPATCH ---
            if len(numpy_buffer) >= SAMPLES_THRESHOLD:
                dispatch_samples = numpy_buffer.copy()
                numpy_buffer = numpy_buffer[-SAMPLES_KEEP_TAIL:]
                dispatch_count += 1
                print(f"[DISPATCH] 🔥 dispatch #{dispatch_count} | {len(dispatch_samples)/SR:.1f}s → Layer 2")
                task = asyncio.create_task(process_numpy_and_send(
                    websocket, dispatch_samples, loop, dispatch_count,
                    transcript_accumulator, full_transcript_segments,
                    voice_state, risk_state, state_lock, pending_tasks
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
            transcript_accumulator, full_transcript_segments,
            voice_state, risk_state, state_lock, pending_tasks
        ))
        pending_tasks.append(task)

    # --- Wait for ALL pending tasks to finish ---
    if pending_tasks:
        print(f"[CONNECTION] ⏳ Waiting for {len(pending_tasks)} pending tasks to finish...")
        await asyncio.gather(*pending_tasks, return_exceptions=True)
        print(f"[CONNECTION] ✅ All tasks finished")

    # --- Flush any remaining transcripts to LLM ---
    pending_transcripts = []
    async with state_lock:
        if transcript_accumulator:
            pending_transcripts = transcript_accumulator.copy()
            transcript_accumulator.clear()

    if pending_transcripts:
        combined_text = " ".join(t["text"] for t in pending_transcripts)
        latest = pending_transcripts[-1]
        batch_ids = [t["dispatch_id"] for t in pending_transcripts]
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

            raw_risk_score = int(fraud_report.get("risk_score", 0))
            raw_warning = str(fraud_report.get("system_logic", ""))
            async with state_lock:
                if raw_risk_score > int(risk_state.get("max_risk_score", 0)):
                    risk_state["max_risk_score"] = raw_risk_score
                    if raw_warning:
                        risk_state["warning"] = raw_warning
                sticky_risk_score = int(risk_state.get("max_risk_score", 0))
                sticky_warning_text = str(risk_state.get("warning", "") or raw_warning)

            full_payload = {
                "transcript": combined_text,
                "voice_label": str(latest["voice_result"]),
                "confidence": float(latest["confidence"]),
                "risk_score": sticky_risk_score,
                "warning": sticky_warning_text,
                "is_spoof": bool(latest["voice_result"] == "Spoof/Deepfake"),
                "final_settlement": False,
            }
            sent = await safe_send_json(websocket, full_payload)
            if sent:
                print(f"[FLUSH LLM] ✅ risk={full_payload['risk_score']} sent")
            else:
                print(f"[FLUSH LLM] WS closed, dropped")
        except Exception as e:
            print(f"[FLUSH LLM] ❌ {e}")

    # --- FINAL SETTLEMENT: run once on full transcript after stop + all processing ---
    async with state_lock:
        full_text = " ".join(full_transcript_segments).strip()
        final_voice_label = str(voice_state.get("voice_result", "Unknown"))
        final_confidence = float(voice_state.get("confidence", 0.0))

    if full_text:
        print(f"[FINAL SETTLE] full transcript | {len(full_text.split())} words → Gemini")
        try:
            llm_raw = await loop.run_in_executor(
                executor, analyze_fraud_intent, full_text, final_voice_label, final_confidence
            )
            fraud_report = {"risk_score": 0, "system_logic": ""}
            try:
                fraud_report = json.loads(llm_raw)
            except json.JSONDecodeError:
                pass

            final_risk = int(fraud_report.get("risk_score", 0))
            final_warning = str(fraud_report.get("system_logic", ""))

            async with state_lock:
                if final_risk > int(risk_state.get("max_risk_score", 0)):
                    risk_state["max_risk_score"] = final_risk
                    if final_warning:
                        risk_state["warning"] = final_warning
                elif not str(risk_state.get("warning", "")) and final_warning:
                    risk_state["warning"] = final_warning

                settled_risk = int(risk_state.get("max_risk_score", 0))
                settled_warning = str(risk_state.get("warning", ""))

            final_payload = {
                "transcript": full_text,
                "voice_label": final_voice_label,
                "confidence": final_confidence,
                "risk_score": settled_risk,
                "warning": settled_warning,
                "is_spoof": bool(final_voice_label == "Spoof/Deepfake"),
                "final_settlement": True,
            }
            sent = await safe_send_json(websocket, final_payload)
            if sent:
                print(f"[FINAL SETTLE] ✅ final risk={settled_risk} sent")
            else:
                print(f"[FINAL SETTLE] WS closed, dropped")
        except Exception as e:
            print(f"[FINAL SETTLE] ❌ {e}")

    # --- Close WS from server side ---
    try:
        await websocket.close()
    except Exception:
        pass

    print(f"[CONNECTION] 🔚 Done | chunks={chunk_count} dispatches={dispatch_count}")