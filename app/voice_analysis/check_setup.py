import ollama
from faster_whisper import WhisperModel
import os

def check_llama_model():
    try:
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': 'Say "Brain is Ready"'}])
        print(f"✅ LLM Check: {response['message']['content']}")
    except Exception as e:
        print(f"❌ LLM Error: {e}")

def check_whisper_model():
    try:
        # Using "tiny" model just for a quick test
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✅ Whisper Check: Model loaded and FFmpeg is working.")
    except Exception as e:
        print(f"❌ Whisper/FFmpeg Error: {e}")

if __name__ == "__main__":
    print("--- System Health Check ---")
    check_llama_model()
    check_whisper_model()