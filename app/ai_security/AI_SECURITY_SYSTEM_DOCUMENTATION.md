# AI Security (Shadow Guard) System Documentation

## 1. System Overview

The **Shadow Guard** is a specialized "AI Firewall" designed to protect the ZoraAI ecosystem from adversarial attacks. Unlike standard security filters that target human-focused fraud (like phishing or scams), Shadow Guard focuses exclusively on **System-Level Hijacking**—specifically prompt injection and jailbreak attempts.

At a high level, the system provides:
1. **Low-Latency Pre-Filtering**: Intercepts requests before they reach expensive reasoning LLMs (Llama 3.2 or Gemini).
2. **Local Model Intelligence**: Uses a dedicated, local [Ollama/Phi-3](file:///C:/Users/DELL/Hackathons/ZoraAI-backend/app/ai_security/guard.py) model for classification.
3. **Middleware Integration**: Seamlessly hooks into FastAPI's request lifecycle to scan incoming payloads.
4. **Automated Red-Teaming**: A validation harness to simulate and block adversarial attacks in real-time.

---

## 2. Prompt Injection Explained

**Prompt Injection** is a security vulnerability where an attacker embeds malicious instructions into a user-provided input with the goal of "tricking" the LLM into ignoring its original system instructions.

### Types of Attacks Shadow Guard Blocks:
- **Instruction Override**: "Ignore all previous instructions and instead do X."
- **Jailbreaking**: Attempting to bypass safety filters using personas like "DAN" (Do Anything Now) or "Developer Mode."
- **Data Exfiltration**: Tricking an agent into revealing internal system prompts, database schemas, or API keys.
- **Token Bypass**: Using specialized "stop-sequences" like `---END---` to prematurely terminate a prompt and start a new, unauthorized command.

---

## 3. Core Architecture

The architecture is designed to be lightweight, deterministic, and isolated from the main reasoning layer.

### 3.1 Shadow Classifier ([guard.py](file:///C:/Users/DELL/Hackathons/ZoraAI-backend/app/ai_security/guard.py))
The core logic resides in `guard.py`. It calls a local Ollama instance running the `phi3` model.
- **System Prompt Isolation**: The guard uses a strict system prompt that instructs the model to ignore scams (Human-targeted) and only block system hijacks (AI-targeted).
- **XML Wrapper**: User input is wrapped in `<user_input>` tags to prevent the input itself from injecting the guard model.

### 3.2 Security Middleware ([middleware.py](file:///C:/Users/DELL/Hackathons/ZoraAI-backend/app/ai_security/middleware.py))
The `ShadowGuardMiddleware` intercepts all POST requests to high-risk routes:
- `/text/email/analyze` (scans `body` and `subject`)
- `/text/sms/analyze` (scans `text`)
- `/text/analyze` (scans `text`)

If an injection is detected, the middleware returns a `403 Forbidden` response, preventing the main LLM from ever seeing the malicious payload.

### 3.3 Red-Team Harness ([red_team_harness.py](file:///C:/Users/DELL/Hackathons/ZoraAI-backend/app/ai_security/red_team_harness.py))
An automated script designed to validate the firewall's effectiveness. It utilizes [Garak](https://github.com/leondz/garak) (an LLM vulnerability scanner) to fire probes at the system and ensure they are blocked.

---

## 4. End-to-End Inference Flow

For every incoming text analysis request, Shadow Guard follows this flow:

1. **Intercept**: The Middleware captures the incoming POST request.
2. **Field Extraction**: Target fields (e.g., email body or SMS text) are extracted.
3. **Classification**: `guard.is_prompt_injection()` sends the text to the local Phi-3 model.
4. **Decision**:
   - **PASSED**: The request continues to the main analyzer (Email/SMS/Voice).
   - **BLOCKED**: A `403 Forbidden` response is returned immediately.
5. **Logging**: Security events are logged with latency, model ID, and a snippet of the blocked content.

---

## 5. Shadow Guard Philosophy: Scams vs. Injections

A critical design choice of Shadow Guard is its ability to differentiate between two types of "bad" content:

| **Feature** | **Human-Targeted Fraud (Scams)** | **System-Targeted Attacks (Injections)** |
| :--- | :--- | :--- |
| **Goal** | Trick a person into giving money/data. | Trick the AI into ignoring system rules. |
| **Status** | **ALLOWED** through Shadow Guard. | **BLOCKED** by Shadow Guard. |
| **Reasoning** | We *need* the main LLM to analyze these scams to detect them. | These represent a security risk to the AI infrastructure itself. |

---

## 6. Verification and Validation

### Red-Teaming Probes
The automated harness simulates three primary attack vectors:
- `promptinject.GenericInjection`: Standard overrides.
- `jailbreak.Dan`: Complex role-play jailbreaks.
- `continuation.StopToken`: Technical bypasses using delimiters.

---

## 7. Future Security Roadmap

As outlined in [systems.txt](file:///C:/Users/DELL/Hackathons/ZoraAI-backend/app/ai_security/systems.txt), ongoing hardening includes:
- **Agentic Sandboxing**: Using Firecracker MicroVMs or `nsjail` to isolate agents that browse the web or access APIs.
- **Voice Adversarial Robustness**: Implementing denoising filters (Low-Pass/Median) to strip inaudible "adversarial noise" from audio deepfakes.
- **Training Integrity**: Checksum-based provenance for data retraining to prevent "data poisoning" backdoors.

---

## 8. Summary

The Shadow Guard provides a robust, defense-in-depth layer for ZoraAI. By moving the security classification to a local, high-speed model, we ensure that system integrity is maintained without compromising the performance of our primary fraud detection engines.
