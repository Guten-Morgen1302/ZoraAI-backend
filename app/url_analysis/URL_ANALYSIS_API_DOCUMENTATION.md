# URL Analysis System Documentation

## Overview
The URL analysis subsystem is a layered, defense-in-depth pipeline that combines static URL intelligence, infrastructure trust checks, browser sandbox behavior telemetry, feature fusion, ML scoring, and optional LLM reasoning.

It is designed so that:
- no single weak signal decides the verdict,
- every layer contributes structured evidence,
- runtime diagnostics can confirm whether each layer executed,
- request and result data are persisted for auditability and feedback loops.

Endpoint:
- POST /url/analyze

Primary module entry points:
- app/url_analysis/router.py
- app/url_analysis/url_analysis.py

## End-to-End Request Lifecycle
1. API receives URL request payload.
2. URL is normalized and validated.
3. Request metadata is persisted in url_analysis_requests with status=processing.
4. Layered analysis pipeline executes through phase 4.
5. ML engine predicts phishing probability and computes composite risk score.
6. Optional LLM reasoner generates human-readable explanation.
7. Response payload is persisted in url_threat_results and request status becomes completed.
8. API returns complete analysis response including pipeline_checks.

On any pipeline failure, the API still returns structured signals where possible and persists failure context.

## Layer-by-Layer Deep Explanation

### Layer 1: URL Lexical and Structural Intelligence
Source module:
- app/url_analysis/feature_extractor.py

What it analyzes:
- raw URL length and token complexity,
- counts of separators and special symbols,
- suspicious token presence (login, verify, auth style keywords),
- IP-as-host usage,
- entropy as a proxy for random/obfuscated URL patterns.

Why it matters:
- phishing links often rely on visual confusion, length inflation, and noisy token composition.
- lexical anomalies are early indicators before network calls occur.

Output shape examples:
- url_length
- num_dots
- num_hyphens
- has_ip
- num_suspicious_keywords
- entropy


### Layer 2: Domain/DNS/WHOIS Infrastructure Intelligence
Source module:
- app/url_analysis/domain_intelligence.py

What it analyzes:
- A records and record count,
- MX and NS presence,
- TTL behavior,
- fast-flux heuristics,
- WHOIS metadata availability (registrar/private registration signals and date fields if available).

Why it matters:
- phishing campaigns frequently use unstable DNS and low-trust infra footprints.
- infra-level anomalies remain useful even when page content appears benign.

Important note:
- WHOIS can be partially unavailable based on provider limits, parser inconsistencies, or registry response quality.
- pipeline_checks distinguishes extraction invocation from live data availability.


### Layer 3: TLS/Transport Trust Intelligence
Source module:
- app/url_analysis/tls_intelligence.py

What it analyzes:
- HTTPS presence,
- certificate validity and self-signed status,
- issuer metadata,
- days to expiry,
- protocol/security header context (for example HSTS presence).

Why it matters:
- transport trust and certificate hygiene provide strong authenticity and operational-quality indicators.
- weak TLS posture can elevate suspicion even if lexical features are mild.


### Layer 4: Homoglyph and Punycode Spoofing Detection
Source module:
- app/url_analysis/homoglyph_detector.py

What it analyzes:
- xn-- encoded domains (punycode),
- mixed-script hostname composition,
- normalized-character mapping to detect visual spoofing,
- brand-similarity heuristics,
- final homoglyph attack flag.

Why it matters:
- this catches visual deception attacks that bypass naive string matching.
- spoofing can occur even with clean TLS and normal redirect flow.


### Layer 5: Browser Sandbox Telemetry Collection
Source module:
- app/url_analysis/sandbox_analyzer.py

Execution model:
- isolated Playwright context, hardened browser launch flags, controlled runtime settings.

What it collects:
- final URL after navigation,
- redirect chain,
- DOM size,
- script count and external JS list,
- network requests and suspicious endpoints,
- cookie and Set-Cookie telemetry,
- sub-analysis outputs for phishing behavior and fingerprint/beacon analysis.

Safety and reliability behavior:
- strict timeout budget,
- resilient partial telemetry return on navigation errors,
- structured error payloads with stage/run identifiers,
- Windows event loop compatibility fallback for Playwright subprocess requirements.

Response policy:
- raw_html is intentionally removed from API response payload to reduce payload size and data leakage risk.

Execution modes:
- local mode (default): sandbox runs Playwright directly in backend process.
- docker mode: sandbox runs in a disposable hardened Docker container and returns JSON telemetry to backend.
- auto mode: attempts docker first, then falls back to local mode if container execution fails.

Environment variables:
- URL_SANDBOX_MODE=local|docker|auto
- URL_SANDBOX_DOCKER_IMAGE=url-sandbox
- URL_SANDBOX_DOCKER_MEMORY=512m
- URL_SANDBOX_DOCKER_CPUS=1.0
- URL_SANDBOX_DOCKER_PIDS=256
- URL_SANDBOX_DOCKER_TIMEOUT_SEC=45


### Layer 6: Redirect/Iframe/CSP Phishing Behavior Analysis
Source module:
- app/url_analysis/phishing_behavior_analyzer.py

What it analyzes:
- redirect depth and cross-domain redirect spread,
- iframe count and external iframe usage,
- login/auth keyword signals in iframe sources,
- CSP posture and risky directives.

Why it matters:
- phishing kits often use deep redirects and embedded content delivery.
- CSP weaknesses frequently correlate with low-security phishing infrastructure.


### Layer 7: Cookie Security and Session Integrity Analysis
Source module:
- app/url_analysis/cookie_analyzer.py

What it analyzes:
- missing Secure/HttpOnly/SameSite attributes,
- cookie scope and insecure cookie patterns,
- cookie risk score normalization,
- optional session fixation checks when before/after-login cookies are available.

Why it matters:
- cookie weaknesses enable session theft, fixation, and CSRF-style abuse.
- cookie posture is a direct application-layer trust signal.


### Layer 8: Fingerprinting and Beaconing Detection
Source module:
- app/url_analysis/fingerprint_beacon_analyzer.py

What it analyzes:
- runtime hooks for canvas/webgl/audio fingerprint patterns,
- navigator.sendBeacon interception,
- suspicious telemetry endpoints,
- external request behavior and beaconing risk classification.

Why it matters:
- many malicious or deceptive sites perform identity fingerprinting and hidden telemetry exfiltration.
- this layer captures behavior invisible to static URL parsing.


### Layer 9: Feature Fusion Engine
Source module:
- app/url_analysis/feature_fusion_engine.py

What it does:
- maps heterogeneous outputs from all prior layers into a fixed schema vector,
- safely converts mixed data types into normalized numeric values,
- enforces deterministic bounds and scaling,
- computes grouped sub-scores for url/infra/content/behavior dimensions.

Why it matters:
- ML requires stable, fixed-shape vectors.
- fusion normalizes cross-layer evidence into a model-consumable representation.


### Layer 10: ML Risk Scoring Engine
Source module:
- app/url_analysis/ml_risk_engine.py

What it does:
- loads persisted model artifacts,
- predicts phishing probability from fused features,
- computes composite risk score and risk band,
- exposes model identity and component-level score breakdown.

Fallback behavior:
- if model artifacts cannot be loaded, heuristic fallback scoring is used.

Why it matters:
- probability and calibrated risk improve ranking and actionability for SOC-style workflows.


### Layer 11: LLM Reasoner (Optional Narrative Layer)
Source module:
- app/url_analysis/llm_reasoner.py

What it does:
- consumes structured evidence and returns concise explainable JSON output.
- returns label, confidence, explanation, key indicators, and recommendations.

Important policy:
- domain age/date fields are excluded from LLM context to avoid age-biased narrative wording.
- this does not remove those fields from technical domain_features returned by the core pipeline.


## Pipeline Integrity Diagnostics (pipeline_checks)
The API includes runtime diagnostics showing whether each layer actually executed:
- phase_1_static_extracted
- whois_extractor_invoked
- whois_has_live_data
- phase_2_tls_extracted
- phase_3_homoglyph_extracted
- playwright_sandbox_invoked
- playwright_sandbox_success
- playwright_network_events
- playwright_redirect_events
- cookie_analyzer_invoked
- phishing_behavior_invoked
- fingerprint_beacon_invoked
- feature_fusion_invoked
- sandbox_error

How to interpret:
- if playwright_sandbox_invoked=true but playwright_network_events=0 and sandbox_error is non-empty, the sandbox started but failed before meaningful navigation telemetry.
- whois_extractor_invoked=true with whois_has_live_data=false means extractor executed but upstream WHOIS data was unavailable/empty.

## Persistence Model

### Table: url_analysis_requests
Purpose:
- stores API-level request envelope and ownership.

Fields:
- id (UUID primary key)
- user_id (nullable UUID from request auth context)
- source_url
- normalized_url
- status (processing/completed/failed)
- created_at


### Table: url_threat_results
Purpose:
- stores analysis payload, prediction payload, and explanation text.

Fields:
- id (UUID primary key)
- request_id (unique FK to url_analysis_requests.id)
- result (serialized API response payload)
- prediction (serialized model output)
- explanation (LLM explanation or fallback)
- created_at

Relationship:
- one request row maps to one result row.

## Error Handling Model
- URL validation errors return 400.
- Pipeline execution exceptions return 500 and mark request status failed.
- Sandbox failures are captured in structured sandbox_error with stage/run context.
- LLM failures degrade gracefully to llm_enhanced=false style output with fallback narrative fields.

## Performance and Payload Hygiene
- sandbox runtime uses bounded timeout windows.
- large HTML content is collected internally for telemetry but excluded from API response output.
- normalized vectors keep downstream ML latency predictable.

## Security and Privacy Considerations
- sandbox runs with restricted browser capabilities and safer defaults.
- no raw_html is sent back to clients in response payload.
- persistence is request/result scoped and supports auditing with user_id linkage.

## Operational Recommendations
- restart server after event-loop policy changes on Windows.
- keep Playwright browser binaries installed and updated.
- retrain or re-export sklearn model artifacts when sklearn major/minor version changes to avoid model compatibility warnings.
- monitor pipeline_checks and sandbox_error in logs for early detection of runtime regressions.
