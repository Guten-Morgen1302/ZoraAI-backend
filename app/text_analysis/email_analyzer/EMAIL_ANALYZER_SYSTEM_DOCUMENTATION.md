# Email Analyzer System Documentation

## 1. System Overview

This Email Analyzer is a multi-signal phishing intelligence pipeline designed to classify risky email content using layered evidence and persist each analysis for auditability.

At a high level, the system combines:

1. Gmail data acquisition (latest mail and by-message lookup)
2. deterministic email preprocessing
3. transformer-based semantic classification
4. stylometry-based writing behavior scoring
5. vector-memory similarity search over known fraud content
6. weighted threat scoring and fraud typing
7. optional LLM explanation enhancement
8. PostgreSQL persistence of request + analysis outputs

The architecture intentionally blends retrieval, ML, and deterministic processing because each component addresses different blind spots:

- preprocessing normalizes noisy sender/subject/body inputs and extracts structured artifacts
- NLP models semantic phishing intent
- stylometry captures manipulation tone and writing behavior
- vector similarity catches known scam families with mutated wording
- weighted scoring produces a stable operational decision
- optional LLM explanation improves analyst readability for difficult cases
- persistence creates traceability and future retraining evidence

This creates a practical production setup where detection quality and explainability are both prioritized.

---

## 2. End-to-End Inference Flow

For every email analysis request, the engine runs an ordered pipeline:

1. Ingest email payload (from Gmail APIs or extension manual payload)
2. Preprocess sender, subject, and body
3. Build normalized text for model consumption
4. Run NLP phishing prediction
5. Run vector similarity search in fraud memory
6. Run stylometry probability scoring
7. Fuse signals into final risk and fraud type
8. Optionally invoke LLM explanation path
9. Persist request and analysis artifacts to PostgreSQL
10. Return structured API response to caller

The final response includes risk, confidence, typed outcome, and evidence sub-scores, with optional LLM fields.

---

## 3. Input Sources and API Surface

The Email Analyzer currently supports these email-specific API routes under /text:

1. /email/latest
2. /email/analyze/latest
3. /email/analyze/by-id
4. /email/analyze/extension

### 3.1 /email/latest

Fetches the latest Gmail message and returns preprocessing output. This route is retrieval-focused and does not run full risk scoring.

### 3.2 /email/analyze/latest

Fetches latest Gmail message, runs full analysis, and returns threat scoring outputs.

### 3.3 /email/analyze/by-id

Fetches a specific Gmail message by message ID/thread context and runs full analysis.

### 3.4 /email/analyze/extension

Accepts direct sender/subject/body payload from the Gmail browser extension and runs full analysis. Supports with_llm_explanation mode.

---

## 4. Gmail Acquisition Layer

The Gmail acquisition client is implemented in gmail_client.py and uses OAuth + Gmail readonly scope.

### 4.1 Auth and Token Behavior

- scope: https://www.googleapis.com/auth/gmail.readonly
- token refresh is supported
- force_reauth can invalidate token file and restart OAuth flow
- local OAuth callback is run on port 7000

### 4.2 Parsing Strategy

- MIME payload is traversed recursively
- prefers text/plain content
- falls back to text/html or snippet when needed
- extracts sender and subject from headers

### 4.3 Failure Handling

Recoverable failures raise GmailClientError, which API layer maps to 502 where appropriate.

---

## 5. Preprocessing Pipeline

The preprocessing layer is implemented in email_preprocessing.py and is used as authoritative normalization before model stages.

### 5.1 Canonical Output

preprocess_email_message returns:

- clean_text
- normalized_text
- original_text
- urls
- sender_domain
- features
- metadata

### 5.2 Metadata Enrichment

Metadata includes sender parsing, URL analysis, safe text truncation fields for LLM, and preprocessing version tags.

### 5.3 Deterministic Guarantees

The implementation emphasizes stable canonicalization so semantically identical email variants produce consistent downstream behavior.

---

## 6. NLP Classifier Layer

The email NLP classifier is implemented in model_inference.py.

### 6.1 Loading and Runtime

- lazy one-time model loading with thread lock
- candidate model dir probing from email_analyzer paths
- validates required artifacts (config + tokenizer config)
- device auto selection with CUDA fallback to CPU

### 6.2 Prediction Contract

predict_email_text returns:

- label
- confidence
- phishing_probability
- risk_score
- model_dir

### 6.3 Label Semantics

Phishing index is resolved using model label metadata when available, with binary fallback behavior.

---

## 7. Stylometry Layer

The stylometry layer is implemented in stylometry.py and backed by a RandomForest model.

### 7.1 Feature Families

Representative features include:

- lexical measures (lengths, diversity)
- punctuation and casing pressure
- URL/domain/email density
- urgency, financial, and social engineering phrase hits
- structural writing cues (sentence shape, repeated punctuation)

### 7.2 Runtime Behavior

predict_stylometry_score returns stylometry_score in [0, 1], compatible with threat scoring fusion.

### 7.3 Training Path

train_stylometry_model supports dataset path overrides and persists model artifacts with feature ordering for stable inference.

---

## 8. Vector Memory Similarity Layer

The similarity layer is implemented in similarity.py using the shared fraud memory embedding service backed by Pinecone.

### 8.1 Search Behavior

Given normalized input text:

1. generate embedding
2. query top-k nearest vectors
3. return similarity score and match context

### 8.2 Output Contract

find_similar_email_messages returns:

- similarity_score
- matched_label
- high_risk
- threshold
- top_k
- matched_text
- matched_source
- top_k_matches

### 8.3 Thread-Safe Caching

The service object is cached with lock protection to avoid repeated expensive initialization.

---

## 9. Threat Scoring Engine

Threat scoring is implemented in threat_scoring.py.

### 9.1 Weighted Fusion

Current fusion formula:

- NLP score weight: 0.4
- Similarity score weight: 0.3
- Stylometry score weight: 0.3

Final score is clamped to [0, 1].

### 9.2 Fraud Typing

Current score mapping:

- >= 0.7 -> email_phishing
- >= 0.4 and < 0.7 -> suspicious_email
- < 0.4 -> safe

### 9.3 Confidence

Confidence is computed from combined score and strongest sub-signal, then bounded in [0, 1].

---

## 10. LLM Explanation Layer

Optional LLM reasoning is implemented in llm_reasoner.py.

### 10.1 Provider and Model

- provider path: local Ollama
- model default: phi3:mini
- client library: direct HTTP requests to Ollama generate endpoint

### 10.2 Prompt Design

Prompt includes sender, subject, body excerpt, and model signals:

- nlp_label
- nlp_score
- similarity_score
- stylometry_score
- final risk score

The LLM is instructed to return strict JSON:

- final_label
- confidence
- explanation

### 10.3 Safety and Reliability

- structured parser with fallback behavior
- normalization of output labels
- timeout/tokens/temperature controls via environment

---

## 11. Persistence Architecture

Email analysis persistence now writes to PostgreSQL for each analyzed email request.

### 11.1 Request Persistence

A row is created in phishing_requests with:

- user_id (if present)
- source = email
- text composed from sender + subject + body

### 11.2 Generic Analysis Metadata

A row is created in phishing_analysis with:

- request_id
- link_count (URL count from preprocessing)
- urgency_score (from preprocessing features)
- status = completed

### 11.3 Email Result Persistence

A dedicated row is created in email_threat_results with:

- request_id
- result (serialized final response fields)
- prediction (serialized NLP prediction)
- explanation (LLM explanation if available, otherwise fallback)

This storage is implemented through PhishingRepository and executed from the email analysis flow.

---

## 12. Database Model Additions

To support email-specific persistence, the following model relationship structure is active:

1. phishing_requests (base request table)
2. phishing_analysis (generic analysis metadata)
3. email_threat_results (email-specific final output table)
4. sms_threat_results (SMS-specific final output table)

PhishingRequest now includes a one-to-one relationship with EmailThreatResult.

---

## 13. Router Integration Details

Email persistence is integrated in _run_email_full_analysis and applies to:

1. /email/analyze/latest
2. /email/analyze/by-id
3. /email/analyze/extension

Each endpoint now resolves request context and DB session, runs analysis, persists artifacts, commits transaction, and returns the API response.

---

## 14. Browser Extension Integration Context

The Gmail extension calls /text/email/analyze/extension with:

- sender
- subject
- body
- with_llm_explanation

These extension-origin analyses are now persisted in PostgreSQL the same way as Gmail API-driven analyses.

---

## 15. Reliability and Failure Handling Philosophy

The implementation follows graceful failure handling:

- Gmail client failures are surfaced as HTTP errors with explicit context
- input validation failures are mapped to 400
- runtime/model errors are mapped to 500
- persistence is committed only after all required artifacts are prepared

This gives predictable behavior for both analysts and extension users.

---

## 16. Practical Strengths of Current Email Stack

1. Multi-signal evidence fusion instead of single-model dependence
2. Optional LLM explanation without forcing expensive path every time
3. Retrieval augmentation for known-phishing family matching
4. Strong preprocessing normalization for consistency
5. End-to-end PostgreSQL persistence for auditability and retraining support

---

## 17. Summary

This Email Analyzer is a production-ready phishing intelligence pipeline that combines:

- Gmail ingestion and manual extension payload support
- deterministic email preprocessing
- semantic NLP inference
- stylometry behavior scoring
- vector-memory similarity retrieval
- weighted threat scoring with typed outcomes
- optional Ollama-based explanation enhancement
- persistent request/result storage in PostgreSQL

It is designed to provide both a robust phishing decision and a defensible evidence trace for operational security workflows.
