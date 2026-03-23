from faster_whisper import WhisperModel
import torch

# Use "float16" for GPU, "int8" for CPU to save memory
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

model = WhisperModel("small", device=device, compute_type=compute_type)

def get_transcript(audio_path_or_bytes):
    print("Running Whisper Transcription")
    segments, info = model.transcribe(audio_path_or_bytes, beam_size=5)
    
    text = " ".join([segment.text for segment in segments])
    return text.strip()