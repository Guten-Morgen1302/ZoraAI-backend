# SMS Analyzer System Documentation

## 1. System Overview

This SMS Analyzer is a multi-signal fraud intelligence pipeline designed to classify risky SMS content using layered evidence instead of relying on a single model prediction.

At a high level, the system combines:

1. deterministic text preprocessing
2. URL threat heuristics
3. transformer-based semantic classification
4. stylometry-based writing behavior scoring
5. vector-memory similarity over known fraud patterns
6. optional local LLM reasoning for difficult or ambiguous cases
7. persisted result + feedback loop for continuous learning

The architecture intentionally mixes machine learning, retrieval, and rule-based signals because each layer solves a different failure mode:

- preprocessing standardizes noisy user text and extracts critical artifacts (URLs, phones, emails)
- NLP handles semantic intent
- stylometry catches social engineering tone and writing signatures
- vector memory catches near-duplicate scam patterns even when wording changes
- LLM adds a narrative reasoner for edge-case interpretation when confidence is low or signals conflict

This creates a practical production setup where precision and explainability are both prioritized.

---

## 2. End-to-End Inference Flow

For every SMS analysis request, the engine executes an ordered pipeline:

1. Input normalization and entity extraction
2. URL risk analysis and urgency estimation
3. NLP semantic prediction with confidence
4. Stylometry score estimation from message writing style
5. Similarity search in Pinecone vector memory
6. Multi-signal weighted scoring and fraud typing
7. Conditional LLM enhancement (or forced LLM explanation mode)
8. Response construction and persistence

The final output always contains the core risk fields and explanation metadata, and can additionally include LLM enhancement flags.

---

## 3. Data Preparation Strategy

### 3.1 Original Dataset Reality

The initial training data existed as spam-vs-ham style datasets from multiple sources. Those labels are useful, but they are often operationally too coarse for fraud investigations where intent-level classes matter more than generic spam.

### 3.2 Intent-Oriented Relabeling Approach

You used a BERT-based intent understanding strategy over spam/ham corpora and converted the practical target into fraud-centric semantics:

- spam-like fraudulent intent examples mapped toward scam-style positive class
- benign ham examples mapped toward safe class

This reframing is important because the production objective is not merely "marketing spam detection" but "fraud/scam/phishing risk understanding."

### 3.3 Unified Training Label Space

The implemented training/inference stack normalizes labels into a binary decision surface for model training:

- safe class (0)
- scam/fraud-like class (1)

Internally, this remains compatible with richer response semantics in downstream scoring and explanation layers.

### 3.4 Incremental Learning with Replay

The retraining workflow supports adding new fraud data while retaining old patterns through replay sampling:

- new dataset rows are loaded and normalized
- an old-data replay subset is sampled in a stratified way
- the model is fine-tuned on mixed data

This prevents catastrophic forgetting and keeps the model aligned with both recent attack patterns and historically important signatures.

---

## 4. Preprocessing Pipeline (Foundational Layer)

The preprocessing layer is shared by all downstream analyzers and enforces strict canonicalization.

### 4.1 Text Normalization

Key operations:

- Unicode NFKC normalization to standardize text representation
- homoglyph normalization for lookalike characters (for example Cyrillic/Greek spoofing into ASCII-safe forms where possible)
- removal/replacement of zero-width and non-breaking whitespace artifacts
- whitespace collapse
- lowercase conversion

Why this matters:

- reduces sparsity for NLP and similarity retrieval
- prevents simple obfuscation tricks from bypassing rules
- mitigates lookalike-domain phishing such as раypal.com-style spoofing
- ensures deterministic behavior for repeated inputs

### 4.2 Structured Artifact Extraction

The preprocessing output includes:

- clean_text
- urls list
- phones list
- emails list

This enables downstream components to operate with explicit structured context instead of repeatedly re-parsing raw strings.

### 4.3 URL Canonicalization

Domain-like patterns and bare web strings are normalized to canonical URL forms when possible (for example prefixing protocol where missing), then deduplicated with order preservation.

### 4.4 Input Quality Gates (Abuse Prevention)

Before expensive model and retrieval stages, SMS input passes strict quality checks:

- max text length capped at 4096 characters (roughly 4KB)
- trivial/low-information inputs are blocked (for example "hi", "hello")
- minimum meaningful-content threshold based on character and word count

This prevents low-effort abuse traffic, noisy model invocation, and unnecessary LLM/retrieval spend.

### 4.5 Why This Layer Is Critical

The system intentionally treats preprocessing as an authoritative source of truth for later components. Without this, the same message could produce inconsistent results depending on tiny punctuation or casing changes.

---

## 5. URL Threat Analysis

The URL analyzer computes a risk score and semantic flags from extracted links.

### 5.0 Strict URL Parsing Sandbox

Every extracted URL is sandbox-normalized before scoring:

- protocol + host canonicalization
- IDNA-safe host handling
- tracking query stripping (utm_*, fbclid, gclid, and related parameters)
- DNS resolution checks for domain volatility and resolution integrity
- internal/private IP detection to prevent SSRF-class downstream fetch risks

If a link resolves to private/internal space or shows unstable fast-flux behavior, dedicated risk flags are emitted.

### 5.1 Detection Signals

The current analyzer checks:

1. suspicious top-level domains
2. excessive URL length
3. IP-based hosts
4. high-entropy / random-looking URL tokens
5. dns_unresolved / fast_flux_suspected conditions
6. internal_ip_blocked for SSRF-sensitive hosts
7. tracking_params_stripped marker for canonical analysis safety

### 5.2 Scoring Method

Each detected URL contributes flags. The combined unique flag set is converted into a bounded risk score in [0, 1].

### 5.3 Why It Works

Fraud campaigns commonly use throwaway domains, random path tokens, shortened spoof links, and urgency redirects. URL heuristics catch this even if NLP confidence is ambiguous.

---

## 6. NLP Classifier Layer

### 6.1 Runtime Inference Behavior

The semantic classifier is loaded lazily once and cached in process memory:

- tokenizer + transformer model discovery from known model directories
- robust artifact validation before use
- device selection (GPU if available, CPU fallback)

The prediction contract is:

- label
- confidence
- raw_confidence
- calibration metadata (temperature, Platt parameters, drift snapshot)

### 6.1.1 Confidence Calibration

Raw classifier confidence is post-processed through calibration:

- temperature scaling on logits
- optional Platt scaling path for binary outputs

The returned confidence is calibrated confidence, not raw softmax confidence.

### 6.1.2 Calibration Drift Monitoring

The inference layer continuously records calibration drift telemetry to model artifact storage:

- running absolute delta between raw and calibrated confidence
- EMA-based drift signal
- thresholded drift alert flag for ops visibility

This provides early warning when the model confidence behavior shifts and calibration retuning is needed.

### 6.2 Model Family and Stability

The training artifacts and scripts use RoBERTa-family sequence classification in current workflows for stable fine-tuning and reliable runtime behavior.

### 6.3 Data Conditioning During Training

The training code applies:

- null filtering
- empty text removal
- label coercion and normalization
- class validation

This prevents poisoned gradients from malformed rows and keeps training objective consistent.

### 6.4 Incremental Trainer Capability

The incremental trainer supports:

- mixed schema column discovery for text and labels
- replay ratio control for old-data rehearsal
- max replay cap
- continuation from previous model snapshots

This is key for rolling model updates as new fraud datasets are added.

---

## 7. Stylometry Layer (Behavioral Forensics)

Stylometry adds a complementary feature family: message writing behavior.

### 7.1 Core Feature Set

Representative features include:

- average word length
- uppercase ratio (caps pressure)
- exclamation count
- urgency keyword count
- message length
- punctuation score
- urgency score

### 7.2 Why Stylometry Helps

Fraud messages often carry social pressure, panic framing, and manipulative urgency tone independent of exact lexical content.

Stylometry catches these behavior signals that semantic classifiers may underweight.

### 7.3 Modeling Choice

A RandomForest classifier is trained on stylometry features and converted into a probability score. This allows non-linear interactions between style features without overfitting to one cue.

### 7.4 Runtime Resilience

If stylometry artifacts are unavailable at inference time, the pipeline gracefully falls back to a heuristic score from urgency and URL risk rather than failing the whole request.

---

## 8. Vector Memory and Similarity Search

### 8.1 Embedding Backbone

The vector layer uses sentence-transformer embeddings (all-MiniLM-L6-v2, 384 dimensions) for semantic retrieval.

### 8.2 Pinecone Store Design

Embeddings are stored in Pinecone namespace fraud_vectors with metadata such as:

- message text
- fraud label
- timestamp
- optional source metadata

### 8.3 Similarity Retrieval

For each incoming SMS:

1. generate normalized embedding
2. query nearest neighbors in vector store
3. return top match score and context

### 8.4 Operational Defaulting

Current service defaults are tuned for practical precision:

- top-k retrieved neighbors bounded for performance
- similarity threshold used to mark high-risk near matches

### 8.5 Why Retrieval Matters

Embedding retrieval catches known fraud families even when attackers mutate wording. It serves as memory-based evidence that complements generalization-based NLP.

---

## 9. Rule/Signal Enrichment

Even with ML and vector layers, lightweight rule signals remain useful for fast explainability and conflict checks.

Current signal extraction includes phrase-level indicators such as urgency and verification language markers.

These features are not the final classifier by themselves, but they improve cross-signal consistency checks and explanation quality.

---

## 10. Threat Scoring Engine

The threat scorer combines core model signals into a single risk score.

### 10.1 Weighted Fusion

Primary weighted structure:

- NLP score weight: 0.4
- Similarity score weight: 0.3
- Stylometry score weight: 0.3

This balances semantic intent, memory match, and behavioral style.

### 10.2 Fraud Typing

After score computation, the system maps into typed outcomes (safe / suspicious / phishing/scam variants) based on thresholds and context.

### 10.3 Explanation and Flags

The scorer emits:

- risk score
- confidence
- fraud type
- diagnostic flags
- explanation text

This provides human-readable reasons rather than only a numeric output.

### 10.4 Conflict-Aware Logic

The engine detects conflicting evidence patterns, for example:

- NLP strongly safe while retrieval/style is highly malicious
- NLP strongly malicious while retrieval/style remains very weak

These cases trigger deeper reasoning escalation.

---

## 11. Local LLM Reasoning Layer

### 11.1 Purpose

The LLM layer is not used as a blanket first-pass classifier. It is used as an intelligence enhancer for ambiguity handling and richer explanations.

### 11.2 Runtime Model

- local Ollama-backed model
- current model target: llama3:8b
- no cloud dependency

### 11.3 Prompt Design

The reasoner prompt includes structured evidence:

- SMS text
- NLP label + confidence
- URL flags
- stylometry score
- similarity score
- rule flags

The model is strictly instructed to return JSON-only output with:

- final_label
- confidence
- explanation

### 11.4 Parser Safety

The parser defends against malformed model output and falls back safely to unknown with an explicit failure explanation if JSON cannot be trusted.

### 11.5 Performance Hardening

The local LLM client includes practical optimizations:

- health check against Ollama tags endpoint
- one-time warm-up per model per process
- bounded generation tokens
- timeout and retry/backoff
- keep-alive and session reuse
- in-memory response cache

This significantly reduces latency spikes and improves reliability under local inference constraints.

### 11.6 Invocation Policy

LLM reasoning is invoked when:

1. confidence is below threshold
2. signal conflict is detected
3. caller explicitly requests LLM explanation mode

This keeps performance predictable while still enabling deep reasoning where needed.

---

## 12. Persistence and Continuous Learning

### 12.1 Result Persistence

Each analysis cycle stores request and analysis artifacts in PostgreSQL for auditability and future retraining support.

Persisted fields include:

- request context
- result payload
- prediction details
- explanation

### 12.2 Automatic High-Confidence Memory Sync

Vector memory writes are server-controlled and no longer exposed as a public user feedback endpoint.

Current behavior:

1. SMS analysis completes normally
2. if model confidence is >= 0.85, embedding is auto-upserted to Pinecone
3. failures in memory sync are logged without breaking primary response flow

This keeps vector memory ingestion protected while still enabling continuous enrichment from strong model signals.

---

## 13. Response Construction and User-Facing Delivery

The final response is built as a structured evidence package rather than a bare verdict.

It includes:

- core risk score and typed outcome
- confidence
- explanation and flags
- sub-scores (NLP / similarity / stylometry)
- similarity context
- LLM enhancement metadata

This design supports:

- analyst trust
- explainable decisioning
- debugging and model governance

In other words, the system returns both decision and reasoning trace.

---

## 14. Reliability and Failure Handling Philosophy

The implementation favors graceful degradation over hard failure:

- missing stylometry model falls back to heuristic estimate
- malformed LLM output falls back to safe parser default
- slow LLM path can be skipped unless required
- retrieval layer handles heterogeneous payload labels robustly

This ensures inference continuity under imperfect runtime conditions.

---

## 15. Practical Strengths of This Architecture

1. Multi-evidence design: less brittle than single-model systems
2. Explainability-first output: operationally useful for analysts
3. Memory-augmented detection: catches pattern reuse campaigns
4. Continuous feedback ingestion: system improves with confirmed cases
5. Local LLM control: zero external data exposure for reasoning layer

---

## 16. Current Scope and Future Expansion Ideas

Current implementation is strong for English-centric SMS fraud scenarios with URL-heavy and urgency-heavy attack patterns.

High-value future enhancements could include:

1. richer multilingual preprocessing and intent normalization
2. explicit calibration monitoring across model versions
3. configurable per-organization scoring weights
4. offline evaluation harness for pre-deployment regression checks
5. background embedding enrichment from confirmed-fraud corpus snapshots

---

## 17. Summary

This SMS Analyzer is a production-oriented fraud intelligence stack that merges:

- deterministic preprocessing
- semantic transformer inference
- behavioral stylometry
- vector memory retrieval
- conflict-aware scoring
- local LLM explanation support
- persistence + feedback-driven improvement

It is designed not only to classify SMS risk, but to provide defensible, explainable, and continuously improving fraud decisions in real operational workflows.
