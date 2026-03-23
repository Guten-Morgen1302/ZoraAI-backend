# Voice Analysis System Documentation

## Overview
The Voice Analysis subsystem is a multimodal, deterministic fraud detection pipeline that performs simultaneous acoustic and linguistic analysis of audio input. It is designed to evaluate both the physical authenticity of the speaker's voice and the malicious intent of their spoken words.

It is designed so that:
- acoustic and linguistic features are evaluated in parallel for low latency.
- temporal and spectral artifacts of AI-generated speech are thoroughly analyzed.
- risk scoring is heavily driven by verifiable textual evidence (zero hallucination policy).
- sophisticated deepfakes that lack detectable synthetic signatures can still be flagged based on explicit malicious intent.

Primary module entry points:
- `app/voice_analysis/router.py`

## End-to-End Request Lifecycle
1. API receives an audio file upload payload (e.g. `.wav`, `.mp3`).
2. The audio file is loaded as a byte stream into memory.
3. An asynchronous ThreadPoolExecutor launches two parallel processes:
   - **Acoustic Inference:** ResNet-BiLSTM model execution for spoof detection.
   - **Linguistic Inference:** Faster-Whisper model execution for transcription.
4. The system awaits both tasks to complete and gathers results.
5. The combined signals (transcript, acoustic label, and acoustic confidence) are fed into the LLM logic analyzer.
6. The LLM acts as a localized, rules-based forensic reasoning layer to generate a final deterministic JSON risk report.
7. API returns the complete multimodal response payload.

## Layer-by-Layer Deep Explanation

### Layer 1: Parallel Transcription Engine
Source module:
- `app/voice_analysis/src/transcription.py`

What it does:
- runs `faster_whisper` (Small model) optimized for hardware (FP16 on GPU, INT8 on CPU).
- converts the raw byte stream into a highly accurate continuous text transcript.
- employs a beam search size of 5 for resilient decoding.

Why it matters:
- The actual spoken intent provides the most reliable indicator of social engineering attacks (e.g., asking for an OTP, urgency, account compromise narrative). 
- It isolates the linguistic features from the physical voice properties.

### Layer 2: Acoustic Feature Extraction and Preprocessing
Source module:
- `app/voice_analysis/router.py`

What it analyzes:
- The raw audio is resampled strictly to 16,000 Hz.
- Peak normalization is applied for volume consistency.
- The waveform is chunked into exact 3-second segments.
- 40-dimensional Mel-Frequency Cepstral Coefficients (MFCCs) are extracted.
- Velocity (delta) and Acceleration (delta-delta) coefficients are computed.
- The resulting tensors are stacked into `[3, 40, 300]` representations and standardize-normalized.

Why it matters:
- By generating 3-channel (MFCC, Delta, Delta-Delta) inputs, the model captures not just the frequency distribution of human speech, but the rate of change of those frequencies over time.
- Padding/clipping to strict 3-second blocks eliminates duration-based classification bias, forcing the model to evaluate the texture rather than the length of the file.

### Layer 3: ResNet-BiLSTM Deepfake Detection Model
Source module:
- `app/voice_analysis/src/voice_model.py`

What it analyzes:
- **ResNet Backbone:** Uses deep 2D convolutional residual blocks to construct feature maps from the spectrograms, identifying fine-grained spectral textures, phase continuity, and synthetic vocoder artifacts.
- **BiLSTM Head:** Flattens the spatial maps and analyzes them as sequential time-series data using a 2-layer Bidirectional Long Short-Term Memory network. 

Why it matters:
- Modern Text-to-Speech (TTS) engines like ElevenLabs produce highly realistic frequency curves but frequently fail to emulate human prosody, micro-pauses, and respiratory pacing.
- The ResNet backbone catches the synthetic "robotic" spectral noise, while the BiLSTM scans forward and backward in time to capture unnatural rhythmic flow.

Output execution:
- The model outputs classification probabilities (0: Real, 1: Spoof) per chunk.
- A final Majority Vote algorithm decides the overall label, averaging confidence scores across the winning class for high stability on longer files.

### Layer 4: Deterministic LLM Fraud Analyzer
Source module:
- `app/voice_analysis/src/fraud_analyzer.py`

What it analyzes:
- Consumes the raw transcript and the acoustic reality label.
- Executes via localized `ollama` (Llama 3.2).
- Follows STRICT grounding rules (Zero Hallucination policy) to extract verifiable malicious keywords and phrases.

Risk Override Logic:
- **Neutrality:** Standard business communications yield low risk (0-2) scores.
- **Benign-Real Grounding:** If the voice is Real and the text is Neutral, risk score is strictly 0.
- **Intent Override:** Even if the acoustic model is fooled ("Real" label), the system overrides the risk score to high if the text contains explicit indicators of attack (e.g., requesting OTPs, PINs, wire transfers).

Why it matters:
- AI models inevitably have false positives and false negatives. This dual-verification layer ensures the system fails safe. An attacker perfectly cloning a voice will still fail the authorization gate if semantic logic flags the context of the conversation as malicious social engineering.

## Setup & Dependency Architecture
- Heavy Python data science ecosystem (`librosa`, `numpy`, `soundfile`).
- Deep Learning (`torch`, `faster-whisper`).
- Localized generative AI (`ollama`).
- External System Dependencies: `ffmpeg` for robust low-level audio codec unwrapping.
- Verification scripts (`check_setup.py`) validate hardware-bound model instantiation.
