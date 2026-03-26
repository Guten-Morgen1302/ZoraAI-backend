import json
import os
from dotenv import load_dotenv

load_dotenv()

# --- OpenRouter / Gemini 2.0 Flash config (same pattern as email analyzer) ---
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "15"))
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "200"))
OPENROUTER_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.1"))


def analyze_fraud_intent(transcript: str, acoustic_label: str, acoustic_conf: float) -> str:
    """Analyze transcript for fraud intent using Gemini 2.0 Flash via OpenRouter.
    
    Returns: JSON string with keys: risk_score, is_fraud, system_logic, red_flags
    """
    prompt = f"""[ROLE]: Objective Forensic Analyst.
[INPUT]: 
- Audio Label: {acoustic_label}
- Audio Confidence: {acoustic_conf:.4f}
- Transcript: "{transcript}"

[TASK]: Analyze the transcript for EXPLICIT evidence of fraud.

[STRICT GROUNDING RULES]:
1. ZERO HALLUCINATION: You may only list "red_flags" that are LITERALLY present as words or phrases in the provided Transcript. 
2. NEUTRALITY: If the transcript is a standard business communication (e.g., policy renewal, appointment reminder) with NO threats or data requests, you must assign a LOW risk_score (0-2).
3. AUDIO WEIGHT: If the Transcript is neutral AND the Audio Model says "Real", the risk_score must be 0.
4. OVERRIDE ONLY ON PROOF: Only override a "Real" audio label if the Transcript contains explicit malicious intent (e.g., asking for OTP, PIN, urgent payment to a 'secure' account).

[OUTPUT SCHEMA]:
Return ONLY valid JSON:
- "risk_score": (int 0-10)
- "is_fraud": (boolean)
- "system_logic": (Reasoning based ONLY on provided text.)
- "red_flags": (List only phrases found IN THE TEXT. If none, return [].)"""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[voice-llm] ERROR: OPENROUTER_API_KEY is not set")
        return json.dumps({"risk_score": 0, "is_fraud": False, "system_logic": "LLM unavailable — no API key", "red_flags": []})

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=OPENROUTER_TIMEOUT,
        )

        print(f"[voice-llm] Calling OpenRouter model={OPENROUTER_MODEL}")

        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": "You are a fraud detection API. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=OPENROUTER_TEMPERATURE,
            max_tokens=OPENROUTER_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        result = str(response.choices[0].message.content or "")
        print(f"[voice-llm] Response received ({len(result)} chars)")
        return result

    except Exception as e:
        print(f"[voice-llm] ERROR: {e}")
        return json.dumps({"risk_score": 0, "is_fraud": False, "system_logic": f"LLM error: {e}", "red_flags": []})