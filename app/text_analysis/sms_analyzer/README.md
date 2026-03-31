# SMS Analyze Deep Dive (`/text/sms/analyze`)

This document explains, in full detail, what happens when your backend receives an SMS payload like:

```json
{
  "text": "Your account is suspended. Verify now at http://example.com",
  "include_llm_explanation": false
}
```

It traces the exact runtime flow from request entry to final `risk_score`, including:

- request validation
- user identity resolution
- preprocessing and feature extraction
- NLP model inference
- stylometry scoring
- vector similarity lookup
- threat score fusion
- optional LLM escalation
- database persistence
- API response construction

---

## 1) Where the flow starts

### Route entrypoint
- `app/text_analysis/router.py`
- Route: `POST /text/sms/analyze`
- Handler: `analyze_sms(payload: SMSAnalyzeRequest, request: Request, db: Session = Depends(get_db))`

### Request schema
Defined in `app/schemas.py`:

```python
class SMSAnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4096)
    include_llm_explanation: bool = False
```

Important meaning:
- `text` cannot be empty at API-schema level.
- max accepted raw length is 4096.
- LLM reasoning is optional and controlled by `include_llm_explanation`.

---

## 2) Identity and ownership handling

Inside route handler, user identity is resolved using:

- `_resolve_user_id_for_api_or_cookie(request, db)`

Resolution order:
1. If `request.state.user_id` already exists, use it.
2. If `Authorization: Bearer <token>` exists, token is treated as API key and validated in `api_keys` table.
3. Else if `access_token` cookie exists, decode JWT and extract subject.
4. Else return `None` (anonymous analysis allowed).

This `user_id` gets attached to persisted `phishing_requests` rows when available.

---

## 3) Service orchestration layer

The route instantiates:

- `SMSFraudAnalysisService(db)` from `app/text_analysis/service.py`

Then calls:

- `service.analyze_sms(text=payload.text, include_llm_explanation=payload.include_llm_explanation, user_id=user_id)`

This service is the central orchestrator.

---

## 4) Text quality gate (hard validation)

Before any expensive model call, the service runs:

- `validate_sms_text_quality(text)` from `app/text_analysis/preprocessing.py`

Checks performed:
1. Input must be `str`.
2. Input must not be blank after whitespace normalization.
3. Length must be <= `MAX_SMS_TEXT_CHARS` (`4096`).
4. Must be meaningful:
   - at least `MIN_MEANINGFUL_SMS_CHARS` alphanumeric chars (`12`)
   - at least `MIN_MEANINGFUL_SMS_WORDS` words (`3`)

If any check fails, `ValueError` is raised.
Route converts it to HTTP 400.

So for `{ "text": "" }`:
- Pydantic `min_length=1` already fails at request parsing.
- Even if bypassed, quality gate fails with `Input text must not be empty`.

---

## 5) First persistence write

After validation, service immediately creates a request record:

- repository: `PhishingRepository.create_request(...)`
- table: `phishing_requests`
- fields: `id`, `user_id`, `text`, `source='sms'`, `created_at`

This happens before full scoring, so every attempted valid analysis gets a request ID.

---

## 6) Preprocessing pipeline (`TextPreprocessingPipeline`)

Service calls:

- `self.pipeline.run(validated_text)`
- implementation in `app/text_analysis/pipeline.py`

It internally executes `process_text(raw_message)` which combines base preprocessing + URL analysis + behavior features + entities.

### 6.1 Base normalization (`preprocess_text`)
From `app/text_analysis/preprocessing.py`:

1. Unicode normalization: `NFKC`
2. Homoglyph normalization:
   - uses `confusable_homoglyphs` if installed
   - fallback static transliteration map if not installed
3. Remove zero-width and non-breaking spaces
4. Collapse multi-whitespace
5. Lowercase conversion

Output includes:
- `clean_text`
- extracted `urls`
- extracted `phones`
- extracted `emails`

### 6.2 URL extraction and canonicalization
`extract_urls(cleaned_text)` in `pipeline.py`:
- supports full URLs, `www.*`, and bare domains
- auto-prefixes missing scheme with `https://`
- strips trailing punctuation
- deduplicates while preserving order
- derives domain list

### 6.3 URL risk scoring
`analyze_urls(urls)` from `app/text_analysis/url_analyzer.py`:

Per URL it does:
- URL parsing/sandboxing
- tracking parameter stripping (`utm_*`, `fbclid`, `gclid`, etc.)
- DNS resolution attempts
- internal/private IP detection
- fast-flux suspicion (many public IPs)
- suspicious TLD check (`xyz`, `top`, `click`, ...)
- IP-host URL check
- long URL check
- random/entropy string check

Returns:
- `url_risk_score` in `[0,1]`, computed as `min(1.0, len(unique_flags)/4)`
- sorted `flags`
- `sanitized_urls`

### 6.4 Urgency scoring
`detect_urgency(cleaned_text)` in `pipeline.py`:
- scans fixed phrase set:
  - `urgent`
  - `immediately`
  - `verify now`
  - `account suspended`
  - `limited time`
- score = `phrase_hits / 5`
- clamped and rounded to 4 decimals

### 6.5 Stylometry feature extraction (feature engineering phase)
`extract_stylometry(raw_text, cleaned_text)` in `pipeline.py` computes:
- `word_count`
- `sentence_count`
- `average_word_length`
- `uppercase_ratio`
- `punctuation_density`
- `emoji_count`

These are generic preprocessing features (not final stylometry model probability yet).

### 6.6 Entity extraction
`extract_entities(cleaned_text)` in `pipeline.py`:
- tries loading spaCy `en_core_web_sm`
- if unavailable: returns empty entity buckets
- otherwise outputs ORG, PERSON, LOCATION lists

### 6.7 Pipeline output object used by SMS service
Service reads these fields from pipeline output:
- `clean_text`
- `url_risk_score`
- `url_risk_flags`
- `urgency_score`
- and link count for persistence

---

## 7) Rule flags from cleaned text

Service adds deterministic phrase flags via `_derive_rule_flags(cleaned_text)`:

Phrase -> rule flag mapping:
- `urgent` -> `urgent_language`
- `verify now` -> `verify_now`
- `account suspended` -> `account_suspended`
- `click` -> `click_link`

These are provided to threat scorer and LLM context.

---

## 8) NLP model inference (`predict_sms_text`)

Called from `app/text_analysis/model_inference.py`.

### 8.1 Model loading strategy
`SMSModelInferenceAPI._resolve_model_dir()` searches in order:
1. `app/text_analysis/sms_model`
2. `app/text_analysis/sms_phishing_model`
3. `app/text_analysis/sms_analyzer/sms_model`
4. `app/text_analysis/sms_analyzer/sms_phishing_model`

A valid directory must contain both:
- `config.json`
- `tokenizer_config.json`

Model/tokenizer load once and cache in memory.

### 8.2 Device and tokenization
- device: CUDA if available, else CPU
- max sequence length: `128`
- `truncation=True`, `padding='max_length'`

### 8.3 Confidence calibration logic
After logits:
1. Raw softmax probabilities computed.
2. Temperature scaling applied (`SMS_CALIBRATION_TEMPERATURE`, default `1.6`).
3. For binary classifier, optional Platt scaling with:
   - `SMS_PLATT_A` (default `1.0`)
   - `SMS_PLATT_B` (default `0.0`)

Prediction payload includes:
- `label`
- calibrated `confidence`
- `raw_confidence`
- calibration metadata

### 8.4 Drift telemetry
Each prediction updates `calibration_stats.json` in model dir:
- running absolute delta raw vs calibrated confidence
- EMA drift
- drift alert using threshold `SMS_CALIBRATION_DRIFT_THRESHOLD` (default `0.2`)

---

## 9) Stylometry model probability (`predict_stylometry_score`)

Service calls:
- `app/text_analysis/sms_analyzer/stylometry.py::predict_stylometry_score(validated_text)`

Behavior:
1. load `stylometry_model.joblib`
2. extract stylometric feature map
3. align features by stored `feature_order`
4. run `RandomForestClassifier.predict_proba`
5. return positive-class probability as `stylometry_score`

### 9.1 Fallback if stylometry artifact unavailable
If stylometry model cannot load or infer (`FileNotFoundError`, `RuntimeError`, `ValueError`), service uses:

```text
fallback_stylometry = min(1.0, 0.7 * urgency_score + 0.3 * url_risk_score)
```

So pipeline still responds, never hard-fails solely due to stylometry artifact issues.

---

## 10) Vector similarity retrieval

Service calls:
- `find_similar_sms_messages(cleaned_text, top_k=3, threshold=0.82)`
- implemented in `app/text_analysis/embedding_service.py`

### 10.1 Embedding backend
Uses `FraudMemoryEmbeddingService` (`app/fraud_memory/embedding_service.py`):
- sentence-transformers model: `all-MiniLM-L6-v2`
- expected embedding size: `384`

### 10.2 Pinecone query
Through `PineconeVectorStore.search(...)` (`app/fraud_memory/pinecone_client.py`):
- namespace default: `fraud_vectors`
- query top-k nearest vectors
- includes metadata: text, label/fraud_label, source, source_file, timestamp

### 10.3 Similarity result shape
Returned to scorer/service as:
- `similarity_score` (best match score)
- `matched_label`
- `high_risk` (`similarity_score >= threshold`)
- `threshold`, `top_k`
- `matched_text`, `matched_source`
- `top_k_matches`

### 10.4 Failure mode handling
`find_similar_sms_messages` wraps all exceptions and returns safe fallback:
- all scores zero
- no matches
- `high_risk=False`

So Pinecone outage does not break SMS analysis endpoint.

---

## 11) Threat scoring and risk fusion

Service calls `score_sms_threat(...)` in `app/text_analysis/threat_scoring.py` with all collected signals.

### 11.1 Input normalization
All numerical scores are clamped into `[0,1]`.

### 11.2 NLP maliciousness transformation
`nlp_score` is derived as:
- if NLP label is malicious (`spam|phishing|scam|fraud|malicious`):
  - `nlp_score = nlp_confidence`
- else:
  - `nlp_score = 1 - nlp_confidence`

### 11.3 Core weighted formula (before LLM)

```text
final_score = 0.4 * nlp_score + 0.3 * similarity_score + 0.3 * stylometry_score
```

This is the primary risk fusion equation.

### 11.4 Flag generation
Flags can include:
- `urgent language` if urgency >= 0.4
- `known scam pattern` if similarity high-risk OR matched label malicious
- `suspicious url` if url_risk_score >= 0.25
- `nlp model indicates <label>` if NLP malicious and confidence >= 0.65

### 11.5 Fraud type buckets (pre-LLM)
- `final_score >= 0.7` -> `sms_phishing`
- `0.4 <= final_score < 0.7` -> `suspicious_sms`
- `< 0.4` -> `safe`

### 11.6 Confidence formula

```text
confidence = (final_score + max(nlp_confidence, similarity_score, stylometry_score)) / 2
```

Then clamped `[0,1]`.

---

## 12) Conditional LLM escalation (`analyze_with_llm`)

The scorer decides whether to call LLM if any condition is true:
1. `include_llm_explanation=True` (forced by API caller), OR
2. `nlp_confidence < 0.7`, OR
3. signal conflict pattern detected:
   - NLP strongly safe but other signals high
   - NLP strongly malicious but other signals weak

### 12.1 LLM request context
`app/text_analysis/llm_reasoner.py` builds a bounded prompt with:
- SMS text
- NLP label/confidence
- URL flags
- stylometry score
- similarity score
- rule flags

### 12.2 Runtime behavior
- local Ollama endpoint (default `http://localhost:11434/api/generate`)
- model configurable by env (default `phi3:mini`)
- warm-up, retries, timeout adaptation, keep-alive session, response cache

### 12.3 LLM output contract
Expected JSON fields:
- `final_label`: `safe|spam|phishing|scam`
- `confidence`: 0..1
- `explanation`: string

### 12.4 LLM effect on score
If parsed label/confidence are valid:
1. compute `llm_score` (malicious confidence or safe inverse)
2. blend score:

```text
final_score = 0.7 * final_score + 0.3 * llm_score
```

3. update confidence as average with llm confidence
4. override fraud type by LLM label mapping:
   - safe -> safe
   - spam -> sms_spam
   - scam -> sms_scam
   - phishing -> sms_phishing

### 12.5 Important implementation detail
`score_sms_threat` always rebuilds `explanation` from deterministic `reason_parts` at the end, so the main `explanation` field is rule-derived. The LLM narrative is returned separately in `llm_explanation`.

---

## 13) Additional persistence writes after scoring

In `SMSFraudAnalysisService.analyze_sms(...)`, after scoring:

1. Create `phishing_analysis` row:
   - `request_id`
   - `link_count`
   - `urgency_score`
   - `status='completed'`

2. Create `sms_threat_results` row:
   - `request_id`
   - `result` (JSON string with scores, flags, sub-signals, similarity payload)
   - `prediction` (JSON string from NLP model)
   - `explanation`

3. `db.commit()` finalizes transaction.

Tables involved:
- `phishing_requests`
- `phishing_analysis`
- `sms_threat_results`

---

## 14) API response shaping

Route returns `SMSAnalyzeResponse` with these key fields:
- `request_id`
- `risk_score`
- `fraud_type`
- `confidence`
- `flags`
- `explanation`
- `llm_enhanced`
- `llm_explanation`
- `nlp_score`
- `similarity_score`
- `stylometry_score`
- `prediction` (raw NLP output payload)
- `similarity` (structured top-k match context)
- `url_risk_score`
- `urgency_score`

So caller gets both final verdict and evidence decomposition.

---

## 15) Error mapping and behavior

### 15.1 HTTP 400
Returned when validation-like errors occur (`ValueError`), examples:
- empty/short/invalid text
- invalid top_k/threshold in similarity endpoint

### 15.2 HTTP 500
Returned for internal model/runtime issues not safely degraded at route level, examples:
- missing transformer artifacts when model inference cannot proceed
- runtime model loading errors

### 15.3 Graceful degradations that do NOT fail endpoint
- stylometry artifact missing -> fallback stylometry formula
- Pinecone similarity unavailable -> zeroed similarity payload
- LLM unavailable -> no LLM enhancement, base scoring still returned

---

## 16) Complete sequence (condensed)

1. Client sends `POST /text/sms/analyze`.
2. FastAPI validates `SMSAnalyzeRequest`.
3. Route resolves user via API key or cookie.
4. Service validates text quality.
5. `phishing_requests` row created.
6. Preprocessing pipeline computes cleaned text + URL/urgency/features.
7. Rule flags are derived from cleaned text.
8. NLP model predicts label + calibrated confidence.
9. Stylometry model predicts `stylometry_score` (or fallback).
10. Vector similarity query returns nearest scam pattern score.
11. Threat scorer fuses signals into `risk_score` + `fraud_type` + `confidence`.
12. Optional LLM escalation may refine score/type and add `llm_explanation`.
13. Analysis + threat rows persisted.
14. API returns structured response with all component scores.

---

## 17) Runtime-relevant files for `/text/sms/analyze`

Primary path:
- `app/text_analysis/router.py`
- `app/text_analysis/service.py`
- `app/text_analysis/preprocessing.py`
- `app/text_analysis/pipeline.py`
- `app/text_analysis/url_analyzer.py`
- `app/text_analysis/model_inference.py`
- `app/text_analysis/sms_analyzer/stylometry.py`
- `app/text_analysis/embedding_service.py`
- `app/text_analysis/threat_scoring.py`
- `app/text_analysis/llm_reasoner.py`
- `app/text_analysis/repository.py`
- `app/models.py`

Vector infra:
- `app/fraud_memory/embedding_service.py`
- `app/fraud_memory/pinecone_client.py`

Schemas:
- `app/schemas.py`

App wiring:
- `app/main.py`

---

## 18) sms_analyzer folder context (what is runtime vs training)

In `app/text_analysis/sms_analyzer/`:
- Runtime-critical now:
  - `stylometry.py`
  - `stylometry_model.joblib`
  - `sms_model/` artifacts (used by central `model_inference.py` directory search)
- Training/support assets:
  - `train_sms_phishing_classifier_1.py`
  - `train_sms_phishing_classifier_2.py`
  - `merged_sms_dataset.csv`
  - `metrics_nlp.txt`
  - `metrics_rf.txt`
  - feedback mechanism scripts under `feeback_mechanism/`

Note: `sms_analyzer/model_inference.py` is currently empty. Runtime inference uses `app/text_analysis/model_inference.py`.

---

## 19) Practical interpretation of `risk_score`

Your final `risk_score` is a bounded fusion score in `[0,1]` where:
- lower values indicate low fraud evidence
- mid values indicate suspicious mixed evidence
- high values indicate consistent malicious evidence across multiple signals

Because the system is multi-signal, a high score can result from multiple moderate indicators combining, not only one extreme model confidence.

---

## 20) Example walk-through

Input:

```json
{
  "text": "URGENT! Your SBI account is suspended. Verify now: http://secure-login-verify.top",
  "include_llm_explanation": true
}
```

Likely process outcome pattern:
1. Text passes quality validation.
2. Preprocessing normalizes case/spacing and extracts URL.
3. URL analyzer flags suspicious TLD + possible random token + maybe long URL.
4. Urgency phrase detector hits `urgent` and `verify now` and maybe `account suspended`.
5. NLP predicts malicious class with high confidence.
6. Stylometry score elevated due urgency/caps/punctuation style.
7. Vector similarity finds close known scam message.
8. Threat scorer computes high weighted score.
9. LLM forced by `include_llm_explanation=true`; may refine type to `sms_phishing`/`sms_scam` and attach narrative explanation.
10. Results persisted and returned as structured payload.

---

## 21) Summary

`/text/sms/analyze` is a robust orchestration endpoint that combines:
- hard input validation
- deterministic feature extraction
- transformer semantics
- stylometric behavior scoring
- memory-based nearest-pattern lookup
- weighted risk fusion
- conditional local LLM reasoning
- full persistence and explainable response payloads

That is exactly how your system turns incoming `{text: "..."}` into a final, auditable `risk_score` and `fraud_type`.
