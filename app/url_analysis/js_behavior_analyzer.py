"""JavaScript behavior analyzer for malware and phishing signal detection."""

from __future__ import annotations

import base64
import math
import re
from collections import Counter
from typing import Any

# Precompiled regex patterns for performance in repeated analysis.
_DANGEROUS_PATTERNS: dict[str, re.Pattern[str]] = {
    "eval": re.compile(r"\beval\s*\("),
    "atob": re.compile(r"\batob\s*\("),
    "document.write": re.compile(r"\bdocument\.write\s*\("),
    "Function": re.compile(r"\bFunction\s*\("),
    "setTimeout": re.compile(r"\bsetTimeout\s*\("),
    "setInterval": re.compile(r"\bsetInterval\s*\("),
}

_STRING_LITERAL_PATTERN = re.compile(r"(['\"])(?P<content>(?:\\.|(?!\1).)*)\1", re.DOTALL)
_BASE64_CANDIDATE_PATTERN = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_HEX_ESCAPE_PATTERN = re.compile(r"\\x[0-9a-fA-F]{2}")
_HEX_LITERAL_PATTERN = re.compile(r"\b0x[0-9a-fA-F]{6,}\b")
_SUSPICIOUS_VAR_PATTERN = re.compile(r"\b_0x[a-fA-F0-9]{4,}\b")
_RANDOM_STRING_PATTERN = re.compile(r"[A-Za-z0-9]{24,}")

_HIDDEN_FORM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"type\s*=\s*['\"]password['\"]", re.IGNORECASE),
    re.compile(r"type\s*=\s*['\"]hidden['\"]", re.IGNORECASE),
    re.compile(r"display\s*:\s*none", re.IGNORECASE),
    re.compile(r"visibility\s*:\s*hidden", re.IGNORECASE),
    re.compile(r"opacity\s*:\s*0\b", re.IGNORECASE),
)

_DOM_MANIPULATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdocument\.createElement\s*\("),
    re.compile(r"\bappendChild\s*\("),
    re.compile(r"\binnerHTML\b"),
)

_DELAYED_EXECUTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsetTimeout\s*\("),
    re.compile(r"\bsetInterval\s*\("),
)

_FUNCTION_PATTERN = re.compile(r"\bfunction\b|=>")


def _normalize_scripts(js_scripts: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize script input to a clean list of non-empty strings."""
    if not js_scripts:
        return []

    normalized: list[str] = []
    for item in js_scripts:
        if not isinstance(item, str):
            continue
        content = item.strip()
        if content:
            normalized.append(content)
    return normalized


def _extract_string_literals(js_scripts: list[str]) -> list[str]:
    """Extract quoted string literals from JavaScript snippets."""
    literals: list[str] = []
    for script in js_scripts:
        for match in _STRING_LITERAL_PATTERN.finditer(script):
            content = match.group("content")
            if content:
                literals.append(content)
    return literals


def calculate_entropy(value: str) -> float:
    """Compute Shannon entropy for a single string."""
    if not value:
        return 0.0

    length = len(value)
    frequencies = Counter(value)
    entropy = 0.0
    for count in frequencies.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    return round(entropy, 4)


def detect_dangerous_functions(js_scripts: list[str]) -> dict[str, Any]:
    """Detect dangerous JavaScript API usage and call count."""
    detected_functions: set[str] = set()
    total_calls = 0

    for script in js_scripts:
        for name, pattern in _DANGEROUS_PATTERNS.items():
            matches = pattern.findall(script)
            if matches:
                detected_functions.add(name)
                total_calls += len(matches)

    return {
        "dangerous_functions": sorted(detected_functions),
        "num_dangerous_calls": total_calls,
    }


def _is_probable_base64(candidate: str) -> bool:
    """Heuristic check for base64-encoded payload-like strings."""
    if len(candidate) < 12 or len(candidate) % 4 != 0:
        return False
    if not _BASE64_CANDIDATE_PATTERN.fullmatch(candidate):
        return False

    try:
        decoded = base64.b64decode(candidate, validate=True)
    except Exception:
        return False

    if not decoded:
        return False

    # Printable ratio filter avoids random binary blobs being marked benignly.
    printable_ratio = sum(32 <= byte <= 126 for byte in decoded) / len(decoded)
    return printable_ratio >= 0.55


def detect_obfuscation(js_scripts: list[str]) -> dict[str, Any]:
    """Detect JavaScript obfuscation indicators."""
    string_literals = _extract_string_literals(js_scripts)

    base64_hits = 0
    long_random_hits = 0
    for literal in string_literals:
        if _is_probable_base64(literal):
            base64_hits += 1
        if _RANDOM_STRING_PATTERN.search(literal) and calculate_entropy(literal) >= 4.2:
            long_random_hits += 1

    joined_scripts = "\n".join(js_scripts)
    hex_hits = len(_HEX_ESCAPE_PATTERN.findall(joined_scripts)) + len(_HEX_LITERAL_PATTERN.findall(joined_scripts))
    suspicious_var_hits = len(_SUSPICIOUS_VAR_PATTERN.findall(joined_scripts))

    indicators = {
        "base64_strings": base64_hits,
        "hex_encoded": hex_hits,
        "suspicious_variable_names": suspicious_var_hits,
        "long_random_strings": long_random_hits,
    }
    obfuscation_detected = any(value > 0 for value in indicators.values())

    return {
        "obfuscation_detected": obfuscation_detected,
        "obfuscation_indicators": indicators,
    }


def detect_hidden_forms(js_scripts: list[str]) -> bool:
    """Detect password and hidden form behavior often used in phishing flows."""
    joined_scripts = "\n".join(js_scripts)
    return any(pattern.search(joined_scripts) for pattern in _HIDDEN_FORM_PATTERNS)


def detect_dom_manipulation(js_scripts: list[str]) -> bool:
    """Detect suspicious dynamic DOM manipulation primitives."""
    joined_scripts = "\n".join(js_scripts)
    return any(pattern.search(joined_scripts) for pattern in _DOM_MANIPULATION_PATTERNS)


def detect_delayed_execution(js_scripts: list[str]) -> bool:
    """Detect delayed execution behavior."""
    joined_scripts = "\n".join(js_scripts)
    return any(pattern.search(joined_scripts) for pattern in _DELAYED_EXECUTION_PATTERNS)


def compute_complexity(js_scripts: list[str]) -> dict[str, Any]:
    """Compute JavaScript complexity metrics and normalized score."""
    joined_scripts = "\n".join(js_scripts)
    lines = [line for line in joined_scripts.splitlines() if line.strip()]
    num_lines = len(lines)
    num_functions = len(_FUNCTION_PATTERN.findall(joined_scripts))

    max_depth = 0
    current_depth = 0
    for char in joined_scripts:
        if char == "{":
            current_depth += 1
            if current_depth > max_depth:
                max_depth = current_depth
        elif char == "}":
            current_depth = max(current_depth - 1, 0)

    # Weight a few dimensions and normalize to [0, 1].
    complexity_raw = (num_lines / 120.0) + (num_functions / 20.0) + (max_depth / 15.0)
    complexity_score = max(0.0, min(round(complexity_raw, 4), 1.0))

    return {
        "num_lines": num_lines,
        "num_functions": num_functions,
        "nesting_depth": max_depth,
        "complexity_score": complexity_score,
    }


def _mean_entropy_for_scripts(js_scripts: list[str]) -> float:
    """Calculate entropy score from extracted strings, with full script fallback."""
    literals = _extract_string_literals(js_scripts)
    if literals:
        entropy_values = [calculate_entropy(item) for item in literals if item]
    else:
        entropy_values = [calculate_entropy(script) for script in js_scripts if script]

    if not entropy_values:
        return 0.0
    return round(sum(entropy_values) / len(entropy_values), 4)


def calculate_risk_score(
    num_dangerous_calls: int,
    obfuscation_detected: bool,
    high_entropy: bool,
    hidden_forms: bool,
    delayed_execution: bool,
) -> float:
    """Compute normalized risk score from weighted indicators."""
    raw_score = 0.0
    raw_score += min(max(num_dangerous_calls, 0), 5) * 2.0
    raw_score += 3.0 if obfuscation_detected else 0.0
    raw_score += 3.0 if high_entropy else 0.0
    raw_score += 3.0 if hidden_forms else 0.0
    raw_score += 2.0 if delayed_execution else 0.0

    # Cap normalization by design maximum (5 dangerous calls + all binary features).
    max_score = 21.0
    return round(min(raw_score / max_score, 1.0), 4)


def analyze_javascript_behavior(js_scripts: list[str] | tuple[str, ...] | None) -> dict[str, Any]:
    """Analyze JavaScript snippets and produce security behavior signals."""
    scripts = _normalize_scripts(js_scripts)
    if not scripts:
        return {
            "dangerous_functions": [],
            "num_dangerous_calls": 0,
            "obfuscation_detected": False,
            "entropy_score": 0.0,
            "hidden_forms": False,
            "dom_manipulation": False,
            "delayed_execution": False,
            "complexity_score": 0.0,
            "js_risk_score": 0.0,
        }

    dangerous = detect_dangerous_functions(scripts)
    obfuscation = detect_obfuscation(scripts)
    hidden_forms = detect_hidden_forms(scripts)
    dom_manipulation = detect_dom_manipulation(scripts)
    delayed_execution = detect_delayed_execution(scripts)
    complexity = compute_complexity(scripts)

    entropy_score = _mean_entropy_for_scripts(scripts)
    high_entropy = entropy_score >= 4.2

    js_risk_score = calculate_risk_score(
        num_dangerous_calls=int(dangerous["num_dangerous_calls"]),
        obfuscation_detected=bool(obfuscation["obfuscation_detected"]),
        high_entropy=high_entropy,
        hidden_forms=hidden_forms,
        delayed_execution=delayed_execution,
    )

    return {
        "dangerous_functions": dangerous["dangerous_functions"],
        "num_dangerous_calls": dangerous["num_dangerous_calls"],
        "obfuscation_detected": obfuscation["obfuscation_detected"],
        "entropy_score": entropy_score,
        "hidden_forms": hidden_forms,
        "dom_manipulation": dom_manipulation,
        "delayed_execution": delayed_execution,
        "complexity_score": complexity["complexity_score"],
        "js_risk_score": js_risk_score,
    }


def run_test_cases() -> dict[str, dict[str, Any]]:
    """Run baseline behavior analyzer test scenarios."""
    test_cases: dict[str, list[str]] = {
        "clean_js": [
            "const total = items.reduce((a, b) => a + b, 0); console.log(total);",
            "function render(){ const el = document.querySelector('#app'); if(el){ el.textContent='ok'; } }",
        ],
        "obfuscated_js": [
            "var _0xabc123='YWxlcnQoZG9jdW1lbnQuY29va2llKQ=='; eval(atob(_0xabc123));",
            "var payload='4f2a9b7c3e1d6f8a4b2c9d7e1f3a6b8c';",
        ],
        "phishing_hidden_form_js": [
            "const f = document.createElement('form'); f.innerHTML = '<input type=\\\"password\\\" style=\\\"display:none\\\">'; document.body.appendChild(f);",
            "document.write('<input type=\\\"hidden\\\" name=\\\"token\\\" value=\\\"x\\\">');",
        ],
        "delayed_execution_js": [
            "setTimeout(function(){ sendData(); }, 5000);",
            "setInterval(() => { beacon(); }, 1000);",
        ],
    }

    results: dict[str, dict[str, Any]] = {}
    for case_name, scripts in test_cases.items():
        results[case_name] = analyze_javascript_behavior(scripts)
    return results


if __name__ == "__main__":
    import json

    print(json.dumps(run_test_cases(), indent=2))