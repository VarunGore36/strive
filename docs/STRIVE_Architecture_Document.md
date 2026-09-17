> **Corrected source document, 2026-09-07.** This copy preserves the supplied design, with corrected NII dimensions/licenses, cold-start semantics and EMA arithmetic. Performance numbers and Vox-Profile dimensionality elsewhere in this source are design assumptions, not measurements of this implementation. The implemented acoustic backend is 62-dimensional; the DSP demo keeps its original 24-dimensional profile. Actual Vox trait dimensions are read from frozen exports. See `ARCHITECTURE.md`, `MODELS.md` and `RESEARCH_STUDY.md` for implementation decisions and experiment requirements. Neural throughput and accuracy must be measured; nothing in this copy certifies the original latency estimates.

# STRIVE — Architecture Document
## Streaming Training-free Real-time Identity-Verification Engine

**Prepared by:** Ankit, Department of Data Science and Engineering, IISER Bhopal  
**Context:** SIH Problem Statement 26104 — AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks  
**Organization:** All India Council for Technical Education (Cyber Security Cell)

---

## 1. Problem Being Solved

Modern generative AI systems — specifically Text-to-Speech (TTS) and Voice Conversion (VC) models — can clone any person's voice from as little as 3–10 seconds of reference audio. Attackers use these clones to impersonate CXOs, government officials, and bank executives over VoIP calls to authorize fraudulent transactions, manipulate employees, or bypass verification procedures.

Conventional defenses — caller ID, manual call-back, voice familiarity — fail entirely against high-quality clones. There is no production system today that can detect a cloned voice in real time, on a live call, without prior enrollment of the speaker's voice.

STRIVE solves this. It processes every second of a live call, builds its understanding of the caller from the call itself, and emits a real-time risk score — all without any training data from the target speaker and without retraining when new TTS systems are released.

---

## 2. Core Design Principles

| Principle | What It Means in Practice |
|---|---|
| **Training-free** | All models are frozen. No weights are updated during deployment. A new TTS system released tomorrow does not break the detector. |
| **Zero prior enrollment** | The system does not need a voice sample of the legitimate speaker stored in advance. It builds the speaker profile live from the call itself. |
| **Real-time streaming** | One risk score is emitted every second. Total pipeline latency per chunk is under 200 ms on a T4 GPU — well within the 1-second stride budget. |
| **Language-aware** | The retrieval database is partitioned by accent cluster (hi-IN, ta-IN, en-IN, etc.), giving Indian-accented speakers sharper nearest-neighbor retrieval. |
| **Zero-day robust** | Because the system uses frozen SSL representations and k-NN retrieval rather than a trained classifier, it generalizes to synthesis methods it has never seen. |

---

## 3. System Architecture — Overview

The pipeline has five sequential stages that execute once per second on every incoming audio chunk:

```
Live Audio Stream
        │
        ▼
 [Stage 0] Audio Ingestion & Streaming Buffer
        │
        ▼
 [Stage 1] Language / Accent Identification  ← runs ONCE at call start
        │
        ▼
 [Stage 2] Dual Feature Extraction (parallel)
      ┌──┴──┐
   Left    Right
  XLS-R   vox-profile + XLSR-53
      └──┬──┘
         │
         ▼
 [Stage 3] Three Parallel Scoring Tracks
  ┌───────┬──────────┬──────────┐
Track 1  Track 2   Track 3
s_global s_session s_coherence
  └───────┴──────────┴──────────┘
         │
         ▼
 [Stage 4] Dynamic Weight Controller + EMA Aggregator → s_risk(t)
         │
         ▼
 [Stage 5] Alert Engine → REST/gRPC API
```

---

## 4. Stage 0 — Audio Ingestion and Streaming Buffer

### What Happens
All incoming audio — regardless of source format — is decoded and resampled to **16 kHz mono PCM** (16,000 samples per second, single channel) using FFmpeg. This is the standard input format for all downstream speech models.

The normalized audio stream flows into a **PyAudio ring buffer** configured as follows:

| Parameter | Value | Reason |
|---|---|---|
| Window size | 2 seconds | Provides enough phoneme content (~15–20 boundaries) for reliable detection |
| Stride | 1 second | Gives 1 Hz detection frequency — one score per second |
| Overlap | 50% | Ensures no boundary artifact falls entirely between two non-overlapping windows |
| Samples per chunk | 32,000 | 2 s × 16,000 Hz |

### Supported Input Sources
- VoIP (SIP/RTP)
- WebRTC (browser-based calling)
- Telephony (G.711, AMR-NB)
- Enterprise collaboration platforms (Zoom, Teams via audio tap)
- Direct microphone capture

### Supported Codecs
G.711 · Opus · SILK · AMR-NB · raw PCM — all decoded to 16 kHz before processing.

---

## 5. Stage 1 — Language and Accent Identification

### Model Used
**`speechbrain/lang-id-voxlingua107-ecapa`**  
Architecture: ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation — Time Delay Neural Network)  
Source: SpeechBrain Hub (public, free)  
Coverage: 107 languages

### What It Does
Takes the **first 5 seconds** of the call and outputs an accent cluster tag — for example `hi-IN` (Hindi, Indian), `ta-IN` (Tamil, Indian), `en-IN` (English, Indian). This tag is used in Stage 3 to route the FAISS retrieval query to the correct language partition.

### Why It Is Necessary
The global retrieval database in Track 1 contains utterances from dozens of languages. Without language conditioning, a Hindi-accented genuine speaker's embedding would be compared against American English and Mandarin nearest neighbors — producing noisy, unreliable retrieval results. Partitioning by language ensures that the k nearest neighbors are acoustically comparable to the query.

### Operational Details
- Runs **exactly once** per call, at call start
- Latency: ~180 ms for 5 seconds of audio
- Output: a string tag that persists for the entire call duration
- Does not run again mid-call — accent is assumed stable

### Novel Contribution
**Language-conditioned FAISS partitioning** — no prior published anti-spoofing system has implemented language-aware retrieval partitioning. The standard NII approach uses a flat, single global database.

---

## 6. Stage 2 — Dual Feature Extraction

Both models in this stage run **in parallel** on every incoming 2-second chunk. Both are **completely frozen** — no parameters are updated, ever. They are used purely as feature extractors.

---

### 6.1 Left Branch — CM Embedding Extractor

**Model:** `nii-yamagishilab/xls-r-2b-anti-deepfake`  
**Base architecture:** `facebook/wav2vec2-xls-r-2B` (2 billion parameters)  
**Source:** HuggingFace (public, CC-BY-NC-SA-4.0 checkpoint license)  
**Post-training:** National Institute of Informatics, Japan — trained on ~74,000 hours of real and fake audio  
**Layers:** 48 transformer layers  
**Language coverage:** 128 languages including Indic languages  
**GPU latency (T4):** ~120–150 ms per 2-second chunk  
**Demo swap:** `nii-yamagishilab/mms-300m-anti-deepfake` (300M params, ~30–40 ms, minor accuracy drop)

#### Processing Pipeline
```
Raw 16 kHz waveform (32,000 samples)
        │
        ▼
XLS-R-2B: 48 transformer layers
        │
        ▼
Frame-level features: [T × D], D=1,024 (MMS-300M) or 1,920 (XLS-R-2B)
(one D-dimensional vector approximately every 20 ms)
        │
        ▼
Mean pooling across all T frames
        │
        ▼
D-dimensional CM embedding (one vector per chunk)
        │
        ▼
Linear classification head → CM score (scalar, 0–1)
```

#### What the CM Embedding Captures
XLS-R was originally a self-supervised speech representation model trained to reconstruct masked audio. When post-trained on real vs. fake audio, its embedding space reorganizes so that real speech and synthetic speech occupy different regions. The backbone-dependent CM embedding encodes the acoustic artifact signature of the chunk — spectral irregularities, phase inconsistencies, vocoder fingerprints — all the characteristics that distinguish synthesized from natural speech.

The CM embedding dimensionality depends on the backbone: 1,024 for MMS-300M or 1,920 for XLS-R-2B.

This embedding is what gets indexed in FAISS and queried for nearest-neighbor scoring in Track 1.

---

### 6.2 Right Branch — Profile and Phoneme Extractor

Two sub-models run here simultaneously:

#### Sub-model A: Phoneme Boundary Extractor
**Model:** `facebook/wav2vec2-xlsr-53-espeak-cv-ft`  
**Parameters:** 300 million  
**Source:** HuggingFace (public)  
**Training:** Fine-tuned on Common Voice with eSpeak phoneme targets  
**GPU latency:** ~15–25 ms per chunk

This model produces frame-level phoneme predictions. We use it not for phoneme transcription but for **phoneme boundary detection** — specifically, we identify the timestamps where one phoneme ends and the next begins, and average the frame-level features within each phoneme segment to produce a **phoneme-level embedding vector**.

For a 2-second chunk this typically yields 15–20 phoneme-level vectors, each capturing the acoustic character of one phoneme unit.

Why phoneme-level rather than frame-level? The RTCFake (2026) paper empirically demonstrated that phoneme-level representations are significantly more stable across VoIP codec compression, noise suppression, and echo cancellation than raw frame-level features. Using phoneme-level embeddings means the Session Profile Store (Track 2) is codec-robust.

#### Sub-model B: Vocal Attribute Extractor
**Model:** `tiantiaf0627/vox-profile-release`  
**Source:** GitHub (public)  
**GPU latency:** ~20–35 ms per chunk  
**Output:** 283-dimensional profile vector

The vox-profile toolkit runs a suite of frozen foundation models to extract five voice quality dimensions:

| Attribute | What It Captures |
|---|---|
| **Pitch (F0)** | Fundamental frequency contour — the melody of the voice |
| **Volume** | Energy envelope — how loud and soft the voice gets |
| **Clarity** | Harmonic-to-noise ratio — breathiness, hoarseness |
| **Rhythm** | Speaking rate, pause patterns, syllable timing |
| **Voice texture** | Formant structure — the resonance characteristics of the vocal tract |

Additionally: estimated age, gender classification, and emotion embedding (valence, arousal dimensions).

All 283 values are concatenated into a single profile vector per chunk. This vector is what gets stored in the Session Profile Store and compared across chunks in Track 2.

---

## 7. Stage 3 — Three Parallel Scoring Tracks

Three scores are computed simultaneously for every chunk. Each answers a different question about the audio.

---

### 7.1 Track 1 — Global Artifact Detector

**Color in architecture: Blue**  
**Question answered:** "Does this audio sound like real or fake speech compared to everything we know globally?"

#### The Database
A pre-built FAISS index stored on disk, partitioned by language cluster. Each partition contains approximately 50,000 utterance embeddings with labels (real=0, fake=1).

**Data sources:**
- **ASVspoof2021-DF** — standard anti-spoofing benchmark dataset (real + synthesized using 100+ TTS/VC systems)
- **DE2024** — real-world deepfakes collected from social media (2024)
- **IndicSpeech** — Indian-accented real speech samples for Hindi, Tamil, Telugu, Bengali

#### FAISS Index Type
`faiss.IndexFlatIP` — exact cosine similarity search (no approximation). "Flat" means every stored vector is compared exactly to the query. "IP" stands for Inner Product, which is equivalent to cosine similarity on L2-normalized vectors.

Retrieval of k=20 nearest neighbors from 50,000 vectors takes **~2–3 ms on CPU**.

#### Scoring Method
```
CM embedding of current chunk
        │
        ▼
FAISS query → top-20 nearest neighbors (filtered by language partition)
        │
        ▼
Each neighbor has a label: real (0) or fake (1)
        │
        ▼
s_global = count(fake neighbors) / k
         = count(fake neighbors) / 20

Example: 15 fake neighbors → s_global = 15/20 = 0.75
```

This is called **ratio-based ensemble scoring** — the method validated in the NII paper as superior to simple majority voting because it produces a continuous score rather than a binary decision.

**Output:** `s_global ∈ [0, 1]` — higher means more likely fake.

---

### 7.2 Track 2 — In-Session Profile Store (SPS)

**Color in architecture: Teal**  
**Question answered:** "Does this audio sound like the same person who was speaking earlier in this call?"

This is the most novel component of STRIVE. No prior published system builds a live speaker profile during the call itself.

#### Step 1: Bootstrap Trust Gate (first 5 seconds)

The Session Profile Store starts empty. Before any profiling can begin, the system must establish a baseline of genuinely-sounding audio.

```
First 5 seconds of call → Track 1 only

If s_global < 0.3:
    → Call provisionally trusted
    → SPS initialised with phoneme vectors + profile vectors
       from these first chunks ✓

If s_global ≥ 0.3:
    → Call starts suspicious
    → SPS stays EMPTY for the entire call ✗
    → Detection relies entirely on Track 1 and Track 3
```

Why this matters: If an attacker uses a cloned voice from the very first second, Track 1 catches it immediately and the SPS never gets poisoned with fake data.

#### Step 2: Session Profile Store (SPS)

**Implementation:** `faiss.IndexFlatIP` — in-memory, per-call, destroyed when call ends.

**Stored per accepted chunk:**
- Phoneme-level embedding vectors (from XLSR-53) — captures how this person pronounces sounds
- 283-d vox-profile vector — captures this person's vocal identity characteristics

**Not stored:** Raw audio, raw waveform, any personally identifiable information. The SPS contains only floating-point feature vectors.

#### Step 3: Within-Session k-NN Scoring

For every new chunk after initialisation:

```
New chunk's phoneme vectors + profile vector
        │
        ▼
Query SPS: k=10 nearest stored vectors by cosine similarity
        │
        ▼
s_session = average cosine similarity to 10 nearest SPS entries
(inverted: low similarity = high fake probability)
```

#### Step 4: Self-Filtering SPS Update Rule

A chunk is added back into the SPS only if **both** conditions are satisfied simultaneously:

```
Condition A: s_global < 0.3   (global detector thinks this chunk is real)
     AND
Condition B: s_session > 0.7  (session profile thinks this chunk matches the caller)

If both → add chunk to SPS
If either fails → discard chunk (do not add to SPS)
```

**Why the dual gate?** A sophisticated attacker might attempt a gradual drift attack — slowly morphing the voice over many minutes so the SPS profile drifts toward the attacker's voice. The dual gate prevents this: the global detector catches gradual spectral shifts, and the session similarity threshold ensures only chunks consistent with the established profile enter the SPS.

**Output:** `s_session ∈ [0, 1]` — higher means current voice is less consistent with the established profile (more likely fake). Returns `None` during cold start (SPS not yet initialized). Missing evidence is masked and available weights are renormalized; using zero would incorrectly dilute global evidence.

---

### 7.3 Track 3 — Phoneme Boundary Coherence

**Color in architecture: Coral**  
**Question answered:** "Is the speech flowing naturally across chunk boundaries, or is there a stitching artifact?"

#### The Physical Insight

Human speech is continuous and co-articulated — the end of one word physically influences the start of the next word through lip, tongue, and jaw movement. This is called **coarticulation**. It is a fundamental property of natural human speech.

TTS systems generate speech in discrete units — a sentence, a phrase, sometimes a chunk boundary. Each generation is internally coherent. But at the seam between two consecutive TTS generations, coarticulation is often broken. The last phoneme of generation N and the first phoneme of generation N+1 were never physically co-articulated — they came from separate synthesis passes.

This produces a tiny but measurable discontinuity in the phoneme embedding space at chunk boundaries.

#### Implementation

```
Chunk t phoneme sequence:   [..., /n/, /d/, /ɪ/, /k/, /ɛ/]
                                                         ↑ last phoneme of chunk t

Chunk t+1 phoneme sequence: [/d/, /ɑː/, /t/, /ə/, ...]
                               ↑ first phoneme of chunk t+1

Compute:
cos_sim = cosine_similarity(embedding(last_phoneme_t),
                            embedding(first_phoneme_t+1))

s_coherence = 1 - cos_sim

→ Real speech:  cos_sim high  → s_coherence low  (smooth transition)
→ TTS stitched: cos_sim low   → s_coherence high (boundary artifact)
```

**Computation cost:** One dot product between two vectors. **Less than 1 millisecond.** No database, no stored state, no model inference at this step — only the phoneme embeddings already computed in Stage 2.

**Why it is robust to VoIP:** The RTCFake (2026) paper showed phoneme-level representations are stable under G.711, Opus, and AMR-NB codec compression. The coherence signal survives the call channel.

**Output:** `s_coherence ∈ [0, 1]` — higher means more discontinuity at the boundary (more likely fake).

---

## 8. Stage 4 — Dynamic Risk Aggregation

### Step 1: Dynamic Weight Controller

The three track scores are combined as a time-varying weighted sum:

```
ŝ(t) = w₁(t) · s_global + w₂(t) · s_session + w₃(t) · s_coherence
```

The weights shift as the call matures:

| Session Age | w₁ (global) | w₂ (session) | w₃ (coherence) | Rationale |
|---|---|---|---|---|
| 0–15 s (cold start) | 0.80 | 0.00 | 0.20 | SPS does not exist yet; rely on global detector |
| 15–60 s (building) | 0.50 | 0.30 | 0.20 | SPS exists but shallow; partial trust |
| > 60 s (mature) | 0.30 | 0.50 | 0.20 | SPS is rich; session profile most discriminative |

**Key insight:** After 60 seconds of a genuine call, the SPS has accumulated enough of this specific speaker's vocal characteristics that it becomes more discriminative than the global database — which contains averaged signals from thousands of speakers. A perfect voice clone will still deviate from the established within-call profile, even if it fools the global model.

The weight schedule shown above is a starting heuristic. In practice, these weights are tunable through ablation experiments on a separate validation split; the test split is reserved for final evaluation.

### Step 2: EMA Aggregator

The weighted score ŝ(t) is smoothed over time using an Exponential Moving Average:

```
s_risk(t) = α · s_risk(t−1) + (1−α) · ŝ(t)

where α = 0.7
```

**What α = 0.7 means in practice:**
- The previous second's risk score contributes 70% to the current score
- The current chunk's raw score contributes 30%
- A single anomalous chunk (cough, codec glitch, background noise) cannot push s_risk into the alert zone alone
- Three consecutive maximum-score updates push s_risk from 0.0 to 1 − 0.6³ = 0.784 — above the 0.75 alert threshold. This matches the three-chunk alert design goal.
- Five consecutive maximum-score chunks push s_risk to approximately 0.83 (1 − 0.7⁵ = 0.83193), breaching the 0.75 alert threshold. From zero, the fourth maximum-score update already reaches 0.7599; five updates are sufficient, not necessary.

This behavior matches how a human expert would reason: one unusual second is noise; a sustained pattern is a signal.

**Output:** `s_risk(t) ∈ [0, 1]` — updated every second, persists for the entire call duration.

---

## 9. Stage 5 — Alert Engine and API

### Three-Tier Alert System

| Tier | Threshold | Actions Triggered |
|---|---|---|
| **Safe** | s_risk < 0.5 | No action. Call continues normally. Score logged silently for audit trail. |
| **Warning** | 0.5 ≤ s_risk < 0.75 | UI prompt: "Verify Caller Identity." Employee asked to use pre-agreed challenge word or question. No transaction blocked yet. SMS/push notification sent to verified contact number. |
| **Alert** | s_risk ≥ 0.75 | Hard block on any pending transaction. Automated SMS to the real executive's registered number: "A suspicious call is in progress claiming to be you." 2FA or supervisor escalation required. Incident ID generated and pushed to SIEM. |

### REST / gRPC API Response

One JSON response is emitted per second per active call:

```json
{
  "timestamp": 1725520060,
  "s_risk": 0.72,
  "alert_level": "warning",
  "track_scores": {
    "s_global": 0.68,
    "s_session": 0.81,
    "s_coherence": 0.61
  },
  "session_age_s": 45,
  "weights": [0.50, 0.30, 0.20],
  "latency_ms": 187
}
```

**Downstream consumers:** Banking applications (core banking middleware), contact centre platforms, enterprise communication tools (Zoom/Teams plugins), telecom operator infrastructure, SIEM platforms (Splunk, IBM QRadar).

---

## 10. Complete Model Registry

All models are publicly available and free. No proprietary or paid model is used anywhere in the pipeline.

| Model | HuggingFace / Source | Parameters | License | Role in Pipeline |
|---|---|---|---|---|
| `nii-yamagishilab/xls-r-2b-anti-deepfake` | HuggingFace | 2 B | CC-BY-NC-SA-4.0 | CM embedding extraction; Track 1 global artifact detection |
| `nii-yamagishilab/mms-300m-anti-deepfake` | HuggingFace | 300 M | CC-BY-NC-SA-4.0 | Demo/edge swap for XLS-R-2B; ~57 ms/chunk on T4 |
| `tiantiaf0627/vox-profile-release` | GitHub | Multiple frozen sub-models | Public | 283-d vocal profile vector; Track 2 session profiling |
| `facebook/wav2vec2-xlsr-53-espeak-cv-ft` | HuggingFace | 300 M | Apache-2.0 | Phoneme boundary detection; Track 2 + Track 3 |
| `speechbrain/lang-id-voxlingua107-ecapa` | SpeechBrain Hub | ~20 M | Apache-2.0 | Language/accent ID; runs once at call start |
| FAISS (`faiss-cpu` / `faiss-gpu`) | Meta / pip | N/A | MIT | Similarity search for global DB and per-call SPS |
| FFmpeg + PyAudio + librosa | System / pip | N/A | LGPL / MIT | Audio ingestion, format decoding, 16 kHz resampling |

### Latency Budget (T4 GPU, per 2-second chunk)

| Component | Hardware | Latency |
|---|---|---|
| XLS-R-2B CM embedding | T4 GPU | ~120–150 ms |
| XLS-R-300M (demo swap) | T4 GPU | ~30–40 ms |
| vox-profile extraction | T4 GPU | ~20–35 ms |
| Phoneme boundary model | T4 GPU | ~15–25 ms |
| FAISS k-NN (k=20, 50k vectors) | CPU | ~2–5 ms |
| EMA + alert logic | CPU | < 1 ms |
| **Total (2B model)** | **T4 GPU** | **~200–220 ms** |
| **Total (300M model)** | **T4 GPU** | **~80–100 ms** |
| **Stride budget** | — | **1000 ms** |
| **Headroom (2B model)** | — | **~780 ms (78%)** |

---

## 11. Novel Contributions

STRIVE contains four contributions not present in any prior published system:

### Contribution 1: Language-Conditioned FAISS Partitioning
The retrieval database is split into per-language partitions. The ECAPA-TDNN accent tag routes each query to the correct partition. Existing anti-spoofing systems — including the NII system that STRIVE builds on — use a flat single-language database. This is the first system to implement accent-aware retrieval for Indian multilingual contexts.

### Contribution 2: In-Session Profile Store (SPS)
No prior real-time anti-spoofing system builds a speaker-specific profile from the live call itself. The SPS gives the system a personalized reference for each call, making it dramatically more sensitive to mid-call voice switching and relay attacks than any globally-trained classifier.

### Contribution 3: Self-Filtering SPS Update Rule
The dual gate — requiring simultaneously low global score AND high session consistency before a chunk enters the SPS — is a novel mechanism for preventing profile poisoning attacks. This is specifically designed to defeat gradual drift attacks where an adversary slowly morphs the voice over many minutes.

### Contribution 4: Inter-Chunk Phoneme Boundary Coherence Score
Measuring coarticulation coherence across consecutive streaming chunk boundaries as a real-time detection signal is not present in any prior work. The RTCFake (2026) paper used phoneme consistency in a different context (offline vs. online versions of utterances for training). Applying it as a streaming inference signal — with sub-millisecond latency — is original.

---

## 12. Key Questions and Answers

**Q: Does it need prior voice recordings of the target speaker?**  
No. The SPS is built from the live call. STRIVE has zero dependency on pre-enrollment.

**Q: Does it work on new TTS systems released after deployment?**  
Yes. Because the detection relies on frozen acoustic representations and k-NN retrieval rather than a trained classifier, new synthesis methods still produce detectable artifacts in the XLS-R embedding space. The NII paper validated this on unseen vocoders.

**Q: What about the first 5–15 seconds of the call?**  
During this cold-start window, only Tracks 1 and 3 are active (s_global and s_coherence). Track 2 is inactive (weight = 0). This is by design: realistic attacks rarely begin in the first 10 seconds, and if the call starts fake from second zero, Track 1 catches it immediately and the SPS never initializes.

**Q: What if the attacker uses a perfect clone that fools Track 1?**  
Track 2 catches it. Even a perfect clone will differ in pitch micro-patterns, rhythm, and voice texture from the genuine speaker's established session profile. The longer a genuine call runs, the richer the SPS becomes and the harder it is for any clone to match the stored profile.

**Q: Does it work on Indian accents?**  
Yes, specifically. The language-conditioned FAISS partitioning addresses the Indian accent retrieval problem directly. XLS-R was pretrained on 128 languages including Indic ones. The MMS-300M variant was pretrained on 1,100+ languages.

**Q: What are the hardware requirements?**  
A GPU is required for real-time operation with the 2B model. A free Google Colab T4 instance is sufficient for demonstration. Production deployment targets cloud T4 instances (~$0.60/hour) or on-premise GPU nodes — the same infrastructure banks already use for fraud detection. A CPU-only deployment is feasible with the 300M model and ONNX export (~300–400 ms per chunk, within the 1-second budget).

**Q: What is the detection accuracy?**  
The base XLS-R-2B anti-deepfake model achieves approximately 0.5% Equal Error Rate (EER) on the ASVspoof2021-DF benchmark — making errors on only 1 in 200 clips under benchmark conditions. STRIVE's three-track ensemble is expected to improve on this for within-call attack scenarios, particularly relay attacks, where the global model alone would not have sufficient discriminative power.

---

## 13. Integration Summary

STRIVE exposes a single REST or gRPC endpoint. Any downstream system consumes the per-second JSON response and acts on the `alert_level` field.

**Integration points:**
- Core banking systems (Finacle, Temenos, Oracle FLEXCUBE) — block transaction on alert
- Contact centre platforms (Genesys, Avaya, Cisco UCCE) — UI prompt on warning
- Enterprise communication tools (Zoom, MS Teams) — in-call notification overlay
- Telecom middleware (IMS, SBC) — call interception on alert
- SIEM platforms (Splunk, IBM QRadar, Microsoft Sentinel) — incident logging

**Privacy compliance:**
- No raw audio is stored at any point
- Only floating-point feature vectors are processed and retained (transiently)
- The SPS is destroyed at call end — zero cross-call data retention
- Feature-only processing aligns with India's Digital Personal Data Protection Act (DPDP Act 2023) and GDPR Article 25 (Privacy by Design)

---

*Document version 1.0 — STRIVE Architecture, Ankit Yadav*
