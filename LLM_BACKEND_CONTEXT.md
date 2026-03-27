# ZoraAI Backend Context Notes (LLM Handoff)

## 1) What This Backend Is
ZoraAI backend is a monolithic FastAPI service that exposes multiple threat-analysis subsystems behind one API:

- Authentication and cookie-based session handling
- Text analysis (SMS + Email phishing/fraud)
- URL phishing analysis (layered static + dynamic + ML + optional LLM)
- Attachment sandbox analysis (static multi-engine)
- Voice analysis (batch upload + realtime websocket)
- Fraud memory/retrieval infrastructure (Pinecone + sentence embeddings)
- Prompt-injection guard middleware for selected text endpoints

Core app bootstrap is in app/main.py.

## 2) High-Level Architecture

### 2.1 App Bootstrap and Middleware
Main entrypoint: app/main.py

- Creates FastAPI app
- Applies CORS for localhost frontend + Chrome extension
- Adds middleware:
  - AuthLoggingMiddleware (auth gate + optional Redis token bucket rate limiting)
  - ShadowGuardMiddleware (prompt-injection checks on selected POST JSON endpoints)
- Calls Base.metadata.create_all(bind=engine) at startup (no migration framework in repo)
- Registers routers:
  - /auth
  - /text
  - /url
  - /attachment
  - /voice
  - /voice/ws

### 2.2 Data Layer
Database setup: app/database.py

- SQLAlchemy engine from DATABASE_URL env var
- SessionLocal dependency via get_db()
- No Alembic migration flow present in repo; schema is model-driven + create_all

### 2.3 Domain Routers
- app/auth/router.py
- app/text_analysis/router.py
- app/url_analysis/router.py
- app/attachment_sandbox/router.py
- app/voice_analysis/router.py
- app/voice_analysis/websocket_router.py

## 3) Authentication and Request Gatekeeping

### 3.1 Auth Strategy
Auth endpoints: app/auth/router.py
Cookie/token logic: app/auth/security.py

- JWT access + refresh tokens issued on signup/login
- Tokens stored in HTTP-only cookies (access_token, refresh_token)
- Password flow:
  - SHA256 first
  - then bcrypt hash

### 3.2 Middleware Auth Rules
Auth middleware: app/middleware/auth_logging.py

Important behavior:
- Most routes require access_token cookie
- Auth-exempt prefixes include /auth, /docs, /redoc, /text/email/analyze/extension, etc.
- user_id is added to request.state when token is valid

Rate limiting:
- Uses RedisTokenBucketLimiter.from_env()
- Keyed by user when authenticated, otherwise client IP
- Returns 429 with rate-limit headers when exhausted

## 4) AI Security Guard
Files:
- app/ai_security/middleware.py
- app/ai_security/guard.py

### 4.1 What It Does
ShadowGuardMiddleware inspects request bodies for prompt injection before route handlers on these prefixes:
- /text/analyze (text)
- /text/sms/analyze (text)
- /text/email/analyze* (body + subject)

### 4.2 Model + Failure Mode
Guard uses local Ollama model (default phi3) via /api/generate.

- If guard says TRUE -> request blocked with 403
- If guard service errors:
  - fail-closed by default (SHADOW_GUARD_FAIL_CLOSED=1) -> block request

## 5) Database Model Map
Source: app/models.py

Main tables:
- users
- phishing_requests (generic text/email/sms request envelope)
- phishing_analysis (link_count, urgency)
- sms_threat_results
- email_threat_results
- confirmed_fraud_cases
- sms_feedback
- email_feedback
- url_analysis_requests
- url_threat_results
- url_feedback
- voice_requests
- voice_analysis
- attachment_requests
- attachment_analysis

Pattern used across subsystems:
- request table stores canonical input/event metadata
- result table stores serialized analysis output (JSON-in-Text columns in many cases)

## 6) Endpoint Inventory (Grouped)

### 6.1 Auth (/auth)
- POST /auth/signup
- POST /auth/login
- GET /auth/me
- POST /auth/logout

### 6.2 Text (/text)
History endpoints:
- GET /text/sms/history
- GET /text/sms/history/{request_id}
- DELETE /text/sms/history/{request_id}
- DELETE /text/sms/history
- GET /text/email/history
- GET /text/email/history/{request_id}
- DELETE /text/email/history/{request_id}
- DELETE /text/email/history

Core + model endpoints:
- POST /text/analyze
- POST /text/model/predict
- POST /text/sms/similarity
- POST /text/sms/analyze

Feedback/retrain:
- POST /text/sms/feedback
- POST /text/sms/feedback/retrain
- POST /text/email/feedback
- POST /text/email/feeback  (intentional typo alias also exists)
- POST /text/email/retrain/feedback

Email fetch/analyze workflows:
- POST /text/email/latest
- POST /text/email/analyze/latest
- POST /text/email/analyze/by-id
- POST /text/email/analyze/extension
- POST /text/email/analyze

### 6.3 URL (/url)
- POST /url/analyze
- POST /url/feedback
- GET /url/history
- GET /url/history/{request_id}
- DELETE /url/history/{request_id}
- DELETE /url/history

### 6.4 Attachment (/attachment)
- POST /attachment/analyze
- GET /attachment/history
- GET /attachment/history/{request_id}
- DELETE /attachment/history/{request_id}
- DELETE /attachment/history

### 6.5 Voice (/voice + /voice/ws)
HTTP:
- POST /voice/analyse
- GET /voice/history
- GET /voice/history/{request_id}
- DELETE /voice/history/{request_id}
- DELETE /voice/history

WebSocket:
- WS /voice/ws/live-protect

## 7) Subsystem Behavior Details

### 7.1 Text Analysis (SMS + Email)
Primary files:
- app/text_analysis/router.py
- app/text_analysis/service.py
- app/text_analysis/pipeline.py
- app/text_analysis/repository.py

Flow (SMS):
1) Validate text quality
2) Preprocess text (URL extraction, urgency, stylometry-friendly features)
3) Run NLP classifier
4) Run similarity search against fraud memory vectors
5) Run stylometry model (fallback heuristic if model missing)
6) Score threat and produce explanation
7) Persist request + analysis + threat result
8) Optional auto-upsert high-confidence predictions to Pinecone memory

Flow (Email):
- Similar multi-signal flow using sender/subject/body, stylometry, similarity, NLP scoring, optional LLM narrative
- Supports Gmail fetch and message-id based analysis endpoints

Important implementation notes:
- Many final payloads are stored as JSON strings in DB Text columns
- repository.py handles persistence and feedback retrieval for retraining
- Threat scoring combines deterministic heuristics + model outputs + optional LLM text

### 7.2 URL Analysis
Primary files:
- app/url_analysis/router.py
- app/url_analysis/url_analysis.py
- app/url_analysis/ml_risk_engine.py
- app/url_analysis/feature_fusion_engine.py

Layered pipeline includes:
- URL lexical/static features
- Domain/DNS/WHOIS intelligence
- TLS intelligence
- Homoglyph detection
- Browser sandbox analysis (Playwright)
- Cookie analysis
- Behavior + fingerprint beacon analysis
- Feature fusion and ML scoring
- Optional LLM explanation

Diagnostics:
- API returns pipeline_checks to show which stages executed successfully
- router includes fallback heuristic scoring if ML engine cannot load

### 7.3 Attachment Sandbox
Primary files:
- app/attachment_sandbox/router.py
- app/attachment_sandbox/README.md

Flow:
1) Uploaded file written to temp file
2) Dynamically imports static pipeline entrypoint (run_static_pipeline)
3) Runs multi-engine static analysis (YARA/ClamAV/ML stack per docs)
4) Optional LLM explanation on top of engine result
5) Persists request + analysis
6) Returns normalized per-engine response

### 7.4 Voice Analysis (HTTP)
Primary file: app/voice_analysis/router.py

Flow:
1) Read uploaded audio bytes
2) In parallel:
   - deepfake/spoof inference via ResNetBiLSTM on MFCC+delta+delta2 chunks
   - transcription via get_transcript()
3) Feed transcript + voice result into fraud intent analyzer
4) Persist voice_request + voice_analysis
5) Return normalized response

### 7.5 Voice Realtime (WebSocket)
Primary file: app/voice_analysis/websocket_router.py

Flow highlights:
- Receives webm chunks
- Decodes to numpy (pydub/ffmpeg)
- Dispatches analysis in rolling windows
- Fast sends partial results, batches transcripts for periodic LLM fraud analysis
- Handles STOP_CAPTURE signal and final flush on close

## 8) Fraud Memory / Vector Infrastructure
Primary files:
- app/fraud_memory/embedding_service.py
- app/fraud_memory/pinecone_client.py

Behavior:
- Embedding model: all-MiniLM-L6-v2 (sentence-transformers)
- Expected vector size: 384
- Stores/searches vectors in Pinecone (namespace default fraud_vectors)
- Used by SMS analysis similarity and feedback/retraining utilities

Operational implication:
- First model load may require network access to Hugging Face unless cached
- Pinecone env vars required for vector operations

## 9) Key Runtime Dependencies and External Services
From requirements.txt and code references:

Core framework/data:
- fastapi, starlette, uvicorn
- sqlalchemy, psycopg2-binary
- redis

ML/AI:
- torch, transformers, sentence-transformers
- faster-whisper, librosa, soundfile, pydub
- spacy
- xgboost, lightgbm

External integrations:
- Ollama (guard + voice fraud analyzer)
- Pinecone
- Playwright + browser runtime
- Google APIs for Gmail endpoints
- ClamAV daemon + YARA runtime

## 10) Environment and Configuration (Observed)
Likely important env variables based on code:

Database/Auth:
- DATABASE_URL
- JWT_SECRET
- JWT_ALGORITHM

AI Security guard:
- SHADOW_GUARD_MODEL
- SHADOW_GUARD_OLLAMA_URL
- SHADOW_GUARD_FAIL_CLOSED

Fraud memory:
- PINECONE_API_KEY
- PINECONE_INDEX_NAME
- PINECONE_HOST

Plus typical external service credentials for Gmail/OpenRouter/Ollama as needed by specific modules.

## 11) Known Quirks / Non-Obvious Details

1) Endpoint typo kept for compatibility:
- /text/email/feeback exists alongside /text/email/feedback.

2) Voice router currently has duplicate imports and mixed style:
- app/voice_analysis/router.py contains duplicated APIRouter/File/UploadFile imports and some unused imports.
- Functional, but refactor candidate.

3) requirements.txt appears UTF-16 encoded:
- Tools assuming UTF-8 may display binary-looking output.

4) Schema creation approach:
- Base.metadata.create_all on app startup means no migration history; DB drift risk in shared envs.

5) Prompt guard fail-closed default:
- If Ollama guard is unavailable, protected text requests can be blocked by design.

6) History-delete endpoints were recently added:
- SMS/Email/URL/Attachment/Voice now support delete one and clear all.

## 12) Suggested Mental Model for Another LLM
When making changes, think in this order:

1) Identify subsystem router and response schema first.
2) Check middleware implications (auth + prompt guard + rate limit).
3) Follow persistence path (request table + result table pairing).
4) Respect existing frontend contracts (history shape, IDs, field names).
5) Treat external services as optional/failure-prone; add safe fallbacks.

## 13) Fast Navigation Pointers
- App startup/middleware: app/main.py
- Auth gate logic: app/middleware/auth_logging.py
- Prompt injection guard: app/ai_security/middleware.py
- Schema contracts: app/schemas.py
- Data model source of truth: app/models.py
- Text API orchestration: app/text_analysis/router.py
- URL API orchestration: app/url_analysis/router.py
- Attachment API orchestration: app/attachment_sandbox/router.py
- Voice HTTP analysis: app/voice_analysis/router.py
- Voice realtime stream: app/voice_analysis/websocket_router.py
- Vector memory services: app/fraud_memory/embedding_service.py

## 14) Current Dev Run Context (Observed)
Typical local command seen in use:
- uvicorn app.main:app --reload

This repo also includes malware_scan as separate infrastructure, but the primary FastAPI backend context above is centered in ZoraAI-backend/app.
