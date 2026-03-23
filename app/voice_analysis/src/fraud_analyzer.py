import ollama
import json

def analyze_fraud_intent(transcript, acoustic_label, acoustic_conf):
    prompt = f"""
    [ROLE]: Objective Forensic Analyst.
    [INPUT]: 
    - Audio Label: {acoustic_label}
    - Transcript: "{transcript}"

    [TASK]: Analyze the transcript for EXPLICIT evidence of fraud.
    
    [STRICT GROUNDING RULES]:
    1. ZERO HALLUCINATION: You may only list "red_flags" that are LITERALLY present as words or phrases in the provided Transcript. 
    2. NEUTRALITY: If the transcript is a standard business communication (e.g., policy renewal, appointment reminder) with NO threats or data requests, you must assign a LOW risk_score (0-2).
    3. AUDIO WEIGHT: If the Transcript is neutral AND the Audio Model says "Real", the risk_score must be 0.
    4. OVERRIDE ONLY ON PROOF: Only override a "Real" audio label if the Transcript contains explicit malicious intent (e.g., asking for OTP, PIN, urgent payment to a 'secure' account).

    [OUTPUT SCHEMA]:
    Return ONLY JSON:
    - "risk_score": (int 0-10)
    - "is_fraud": (boolean)
    - "system_logic": (Reasoning based ONLY on provided text.)
    - "red_flags": (List only phrases found IN THE TEXT. If none, return [].)
    """

    print("Running LLM")

    response = ollama.chat(
        model='llama3.2', 
        format='json', 
        messages=[
            {'role': 'system', 'content': 'You are a fraud detection API. Output JSON only.'},
            {'role': 'user', 'content': prompt}
        ]
    )
    
    return response['message']['content']