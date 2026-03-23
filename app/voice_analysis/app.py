import torch
import librosa
import numpy as np
import io
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File

# Custom modules
from src.transcription import get_transcript
from src.fraud_analyzer import analyze_fraud_intent  
from src.voice_model import ResNetBiLSTM 
app = FastAPI()
executor = ThreadPoolExecutor(max_workers=3)

# --- CONFIG ---
SR = 16000
DURATION = 3
N_MFCC = 40
MAX_LEN = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = ResNetBiLSTM().to(device)

model.load_state_dict(torch.load("models/model.pth", map_location=device))
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

# Main Endpoint
@app.post("/voice/analyse")
async def detect_fraud(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    loop = asyncio.get_event_loop()
    
    voice_task = loop.run_in_executor(executor, run_voice_model_logic, audio_bytes)
    transcript_task = loop.run_in_executor(executor, get_transcript, io.BytesIO(audio_bytes))
    
    voice_res, transcript_text = await asyncio.gather(voice_task, transcript_task)
    
    if "error" in voice_res:
        return voice_res

    # Fraud Analysis via LLM
    llm_output_raw = analyze_fraud_intent(
        transcript_text, 
        voice_res['result'], 
        voice_res['confidence']
    )
    
    try:
        fraud_report = json.loads(llm_output_raw)
    except:
        fraud_report = {"error": "JSON parsing failed", "raw": llm_output_raw}

    return {
        "filename": file.filename,
        "voice_analysis": voice_res,
        "transcript": transcript_text,
        "fraud_report": fraud_report 
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)