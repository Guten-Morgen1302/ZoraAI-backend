## 🎥 Demo Video

[![Watch the Demo](https://img.youtube.com/vi/3FK3UoyiKZI/maxresdefault.jpg)](https://www.youtube.com/watch?v=3FK3UoyiKZI)

# Zora Backend Architecture Guide

This document is a deep technical orientation guide for the backend inside the app folder.

Its purpose is to help a new engineer understand:

- what each module is responsible for,
- how modules collaborate end to end,
- where data is stored and why,
- how analysis pipelines are composed,
- which parts are synchronous, asynchronous, or background-driven,
- where model logic, security logic, and persistence logic live.

This document intentionally focuses on architecture and module responsibilities, not endpoint-by-endpoint route documentation.

---

## System at a glance

The backend is a multi-domain fraud analysis platform built on FastAPI + SQLAlchemy and organized as a set of domain subsystems:

- Identity and access foundation (auth, API keys, request-level auth middleware)
- Text intelligence (SMS and email fraud analysis)
- URL intelligence (layered static + behavioral + ML + optional LLM explanation)
- Attachment intelligence (multi-engine static malware analysis)
- Voice intelligence (acoustic spoof detection + transcript intent scoring)
- AI safety firewall (prompt injection defense)
- Fraud memory and retrieval (embedding + Pinecone vector store)
- Portal state (persisted chat sessions)

The architecture is intentionally layered and compositional.
No single heuristic or model is trusted alone for high-stakes fraud decisions.

---

## Runtime composition

### Bootstrapping and application composition

Primary composition happens in main.py:

- FastAPI app creation
- CORS policy setup for local web and extension clients
- middleware registration
- SQLAlchemy metadata bootstrap
- registration of domain routers from all subsystems

Main runtime traits:

- event-loop policy compatibility for Windows is explicitly configured,
- central logging is configured once at application startup,
- middleware-based auth and logging runs before domain handlers,
- the app acts as an orchestration shell, while domain logic remains in modules.

### Shared infrastructure

database.py provides:

- environment-driven SQLAlchemy engine construction,
- pooled DB sessions,
- shared declarative Base.

This module is the persistence root for every subsystem.

---

## Shared data model and persistence strategy

The platform persists nearly all analysis requests and outputs.
This provides auditability, explainability traceability, feedback-loop readiness, and analytics over time.

### Core entity families

models.py defines data entities in these groups:

1. Identity and credentials
- users
- api_keys

2. Text analysis lifecycle
- phishing_requests
- phishing_analysis
- sms_threat_results
- email_threat_results
- sms_feedback
- email_feedback
- confirmed_fraud_cases

3. URL analysis lifecycle
- url_analysis_requests
- url_threat_results
- url_feedback

4. Voice analysis lifecycle
- voice_requests
- voice_analysis

5. Attachment analysis lifecycle
- attachment_requests
- attachment_analysis

6. Portal workspace state
- portal_chats

### Persistence design pattern used repeatedly

Most domains follow the same state-machine style lifecycle:

1. Create a request row with processing status
2. Execute pipeline
3. Save structured prediction/result payloads
4. Mark status as completed or failed

This pattern is visible across text, URL, voice, and attachment systems.

---

## Module-by-module architecture

## 1) auth module

Files:

- auth/security.py
- auth/router.py

Responsibility:

- password hashing and verification,
- JWT creation and validation,
- cookie-oriented auth token handling,
- principal extraction for request-scoped identity.

Security mechanics:

- password hash flow uses SHA-256 pre-hashing plus bcrypt storage,
- JWT supports typed tokens (access and refresh) with expiries,
- helper methods support reading and validating token subject/type.

How it fits:

- middleware and feature routers depend on auth output,
- request identity propagates through request state.

---

## 2) api_keys module

Files:

- api_keys/router.py

Responsibility:

- user-scoped API key lifecycle management,
- generation, masking, activation metadata, and reveal behavior.

Design behavior:

- generated keys are unique and prefixed,
- key records include expiry and revocation metadata,
- key auth is supported in middleware and text paths.

How it fits:

- complements cookie auth for automation or non-browser clients,
- middleware resolves either bearer key flow or cookie flow.

---

## 3) middleware module

Files:

- middleware/auth_logging.py
- middleware/rate_limiter.py

Responsibility:

- central request identity resolution,
- auth context logging,
- rate-limit enforcement with fail-open resilience.

Rate limiting model:

- Redis-backed token bucket algorithm,
- refill and consumption performed atomically,
- request headers enriched with rate-limit telemetry.

Auth resolution behavior:

- checks bearer API key path,
- falls back to access-token cookie path,
- attaches resolved identity to request state.

Why this matters:

- domain modules stay focused on analysis logic,
- cross-cutting concerns remain centralized.

---

## 4) portal module

Files:

- portal/router.py

Responsibility:

- persisted chat workspace state for portal clients.

Data model style:

- chat metadata and message arrays are stored in portal_chats,
- message payloads are serialized JSON,
- user scoping is applied when identity is present.

Why it exists:

- provides durable conversational context independent of analysis engines,
- supports client-side session continuity.

---

## 5) text_analysis module

Core files:

- text_analysis/pipeline.py
- text_analysis/preprocessing.py
- text_analysis/service.py
- text_analysis/repository.py
- text_analysis/threat_scoring.py
- text_analysis/model_inference.py
- text_analysis/embedding_service.py
- text_analysis/llm_reasoner.py

Submodules:

- text_analysis/email_analyzer/*
- text_analysis/sms_analyzer/*

### Architectural role

This is the highest-throughput fraud analysis subsystem for human messages.
It combines deterministic preprocessing, model predictions, similarity retrieval, stylometry, weighted scoring, and optional LLM explanation.

### Pipeline architecture

The SMS and email flows are not single-model predictions.
They are orchestrated fusion pipelines:

1. Normalize and sanitize text
2. Extract structured cues (URLs, urgency, entities)
3. Run semantic model inference
4. Run stylometry risk estimator
5. Run similarity retrieval over fraud memory vectors
6. Fuse scores into risk, confidence, fraud type, and flags
7. Optionally enhance explanation using LLM reasoner
8. Persist result and intermediate payloads

### Service-repository split

service.py orchestrates business flow and scoring composition.
repository.py encapsulates DB writes/reads for request and threat-result entities.

This separation keeps orchestration logic testable while minimizing DB coupling.

### Continuous-learning hooks

The text module includes feedback-aware components:

- feedback tables for corrected labels,
- retraining scripts in email and SMS analyzer folders,
- confirmed fraud case ingestion into vector memory.

This means the system was designed with post-deployment learning in mind.

---

## 6) url_analysis module

Core files:

- url_analysis/url_analysis.py
- url_analysis/router.py
- url_analysis/feature_extractor.py
- url_analysis/domain_intelligence.py
- url_analysis/tls_intelligence.py
- url_analysis/homoglyph_detector.py
- url_analysis/sandbox_analyzer.py
- url_analysis/phishing_behavior_analyzer.py
- url_analysis/fingerprint_beacon_analyzer.py
- url_analysis/cookie_analyzer.py
- url_analysis/feature_fusion_engine.py
- url_analysis/ml_risk_engine.py
- url_analysis/llm_reasoner.py

### Architectural role

This subsystem implements a layered, defense-in-depth URL intelligence stack.
It is one of the most compositional pipelines in the backend.

### Phase model

url_analysis.py composes analysis through explicit phases:

- Phase 1: lexical URL + domain intelligence
- Phase 2: transport/TLS intelligence
- Phase 3: homoglyph and spoofing intelligence
- Phase 4: async browser sandbox behavior intelligence

Then:

- feature fusion produces stable model-ready vectors,
- ML risk engine produces phishing probability and risk score,
- optional LLM reasoner produces analyst-friendly narrative.

### Strong architecture properties

- explicit pipeline checks and execution diagnostics,
- resilience to partial failures (sandbox/whois/network variance),
- heavy telemetry from browser behavior while preserving controlled response shape,
- fallback risk behavior if model artifacts fail.

### Short-link handling

router-level orchestration includes short URL expansion and normalized-target analysis so disguised shorteners are evaluated on destination behavior, not only entry URL shape.

### Why this subsystem is strong

It intentionally integrates static, infrastructural, and behavioral signals before final scoring, reducing sensitivity to any one evasion tactic.

---

## 7) attachment_sandbox module

Core files:

- attachment_sandbox/router.py
- attachment_sandbox/llm_reasoner.py
- attachment_sandbox/app/static_analysis/pipeline.py
- attachment_sandbox/app/static_analysis/* parsers/scanners
- attachment_sandbox/thrember/*

### Architectural role

This module performs static attachment threat analysis with multi-engine synthesis.
It is designed for API-safe and relatively low-latency file inspection without runtime detonation.

### Engine composition

The static pipeline combines:

1. YARA scan
2. ClamAV scan
3. ML score path

ML path strategy:

- PE-focused path via Thrember features/model for executable analysis,
- fallback classifier path for non-PE or unsupported feature contexts.

### Background execution pattern

attachment_sandbox/router.py uses queued/background style processing:

- request row is created with queued status,
- analysis job executes independently,
- status transitions through processing stages,
- final analysis payload is persisted.

This architecture avoids blocking client request cycles on heavier file processing.

### Explainability extension

Optional LLM explanation metadata is attached to final features payload when requested.

---

## 8) voice_analysis module

Core files:

- voice_analysis/router.py
- voice_analysis/websocket_router.py
- voice_analysis/src/voice_model.py
- voice_analysis/src/transcription.py
- voice_analysis/src/fraud_analyzer.py

### Architectural role

This module is a multimodal fraud analyzer for uploaded or streamed audio.
It combines acoustic authenticity signals and linguistic intent analysis.

### Core inference structure

For uploaded audio flow:

- run acoustic spoof model and transcription in parallel,
- fuse transcript + acoustic signal into fraud intent analyzer,
- persist both request artifact and analysis artifact.

For streaming flow:

- websocket pipeline batches chunk processing,
- parallel chunk inference and transcription,
- quick-send partial outputs,
- batched LLM interpretation for accumulated transcript windows.

### Model and signal layers

- acoustic model: ResNet-BiLSTM with MFCC + delta + delta-delta features,
- linguistic model: Whisper-style transcription layer,
- forensic reasoning layer: LLM fraud analyzer guided by strict prompt constraints.

### Why this architecture matters

A purely acoustic system misses intent; a purely transcript system misses deepfake voice risk.
Combining both increases practical robustness.

---

## 9) ai_security module

Core files:

- ai_security/guard.py
- ai_security/middleware.py
- ai_security/red_team_harness.py

### Architectural role

This is an AI firewall subsystem that protects upstream LLM-facing flows from prompt injection and jailbreak attempts.

### Design principles

- classify system-level attack intent before expensive downstream reasoning,
- separate scam-content analysis from model-hijack detection,
- block only AI-targeted prompt abuse, not user-content fraud artifacts.

### Runtime integration

middleware.py intercepts selected high-risk text analysis workloads and applies guard.py classification before normal processing continues.

### Validation posture

red_team_harness.py exists for active adversarial testing using known probe categories.

---

## 10) fraud_memory module

Core files:

- fraud_memory/embedding_service.py
- fraud_memory/pinecone_client.py
- fraud_memory/pipelines/*
- fraud_memory/run_ingest_*.py
- fraud_memory/upsert_csv_to_pinecone.py

### Architectural role

This module is the long-term semantic memory plane for known fraud patterns.
It enables retrieval-augmented risk scoring in text analysis.

### Vector-store architecture

- embeddings are generated for fraud texts,
- vectors are stored in Pinecone namespace(s),
- retrieval returns nearest-known fraud context for similarity scoring.

### Pipeline families

fraud_memory/pipelines includes ingestion flows for:

- text scam datasets,
- email scam datasets,
- URL phishing datasets.

### Why this subsystem is important

It catches near-duplicate or paraphrased fraud patterns that may evade rigid rules or model confidence thresholds.

---

## 11) schemas module

File:

- schemas.py

Responsibility:

- Pydantic contracts for request/response payloads and typed objects,
- stable API surface boundaries independent of internal model classes.

Why it matters:

- creates explicit compatibility boundaries,
- allows internal implementation to evolve while preserving external contract shape.

---

## Cross-module interaction map

### End-to-end request path pattern

Most analysis requests pass through this shared architecture envelope:

1. Middleware resolves identity/auth context
2. Domain-specific orchestrator validates and normalizes input
3. Multi-signal analysis pipeline executes
4. Threat result is synthesized
5. Result and metadata are persisted
6. Structured response is returned

### How subsystems depend on each other

- text_analysis depends on fraud_memory for similarity retrieval,
- middleware depends on auth and api_keys for principal resolution,
- ai_security can front-run text_analysis execution,
- shared models/schemas/database underpin all modules,
- portal remains mostly orthogonal, using shared identity and DB only.

---

## Processing styles used across the backend

### Synchronous orchestration

Used where compute cost is moderate and deterministic request-response timing is acceptable.

Examples:

- many text and URL orchestration paths after preprocessing.

### Async plus parallelized tasks

Used where independent operations can run concurrently to cut latency.

Examples:

- voice acoustic inference and transcription,
- URL browser telemetry collection.

### Background job style

Used for heavier or staged workloads that should not block immediate request cycle.

Examples:

- attachment static pipeline and optional explanation stage.

---

## Reliability and failure-handling design

Common patterns visible in codebase:

- safe JSON decode wrappers around stored payloads,
- explicit status fields for request lifecycle tracking,
- fallback scoring paths when model artifacts are missing,
- partial-result handling for external dependency volatility,
- graceful rollback and cleanup for background job failures.

These patterns indicate production-oriented resilience over “perfect-path-only” design.

---

## Security architecture summary

Security is layered, not centralized into a single gate.

Key layers include:

- auth token and API key identity resolution,
- request-level rate limiting and logging,
- prompt injection firewall for LLM-exposed workloads,
- sandboxed URL behavior analysis,
- static malware scanning for attachments,
- explicit cookie/TLS/infrastructure trust checks in URL pipeline.

This layered model reduces single-point security blind spots.

---

## Model and AI architecture summary

The backend uses multiple model families for different evidence types:

- transformer classifiers for text semantics,
- stylometry models for behavior signatures,
- vector similarity retrieval over Pinecone for memory-based matching,
- ML fusion/scoring models for URL risk aggregation,
- acoustic deep model for voice spoofing,
- local/remote LLM reasoning for explanation and difficult-case interpretation,
- local lightweight guard model for prompt-injection filtering.

This is intentionally a hybrid AI system, not a single monolithic model service.

---

## New engineer onboarding path

Recommended reading/order for fastest understanding:

1. Start with main.py and database.py for composition and infra baseline
2. Read models.py and schemas.py to understand data contracts and persistence entities
3. Read middleware to understand auth context and request lifecycle
4. Read one analysis domain end to end (text_analysis first is easiest)
5. Then read url_analysis for full layered pipeline architecture
6. Read attachment_sandbox and voice_analysis for asynchronous/multimodal patterns
7. Read ai_security and fraud_memory to understand defensive and retrieval layers

---

## Practical mental model of the backend

A useful way to think about this backend:

- It is an orchestration platform for multiple specialized fraud detectors.
- Persistence is a first-class design concern, not an afterthought.
- Pipelines are intentionally redundant so weak signals can be corrected by stronger independent signals.
- Explanation and auditability are treated as core product features.
- Feedback and retraining hooks indicate a continuous-improvement architecture.

In short: this is a layered fraud intelligence backend, not a simple CRUD API with a single classifier.
