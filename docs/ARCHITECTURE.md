# Implemented STRIVE architecture

## SIH presentation architecture (2026-09-14)

The presentation path extends the original API without replacing its scoring
architecture. Microphone frames, decoded uploads and deterministic scenarios all
enter `Call.ingest`/`Call.drain`, the same ring buffer, capture queue, evidence,
fusion and policy path. File replay is controlled over the primary WebSocket and
supports audio-time realtime pacing or accelerated QA. Upload bytes and normalized
audio remain transient and are erased after playback or call teardown.

The 16 kHz canonical contract remains. `config/presentation.json` selects a 2 s
analysis window and 0.5 s hop for approximately two UI updates per audio second.
The original 1 s default remains backward-compatible. Branch scheduling uses the
audio timeline. A gap, silence boundary or capture overflow invalidates continuity
and scheduler caches, so evidence from before an interruption is not presented as
current.

Risk separation is enforced at the policy boundary:

```text
artifact + session + coherence -- reliability mask --> AUTHENTICITY RISK
transaction metadata -------------------------------> CONTEXT RISK
authenticity + context -- policy only --------------> DECISION RISK
```

Channel quality applies one common reliability mask to acoustic branches. Common
scaling cancels during normalization, so a merely degraded channel cannot lower or
raise a valid authenticity score. If quality is unusable, fusion abstains. A future
branch-specific rule requires held-out speech/codec validation before adoption.

Every branch now exposes availability, freshness, age, reliability and a reason.
The three legacy track scores remain for clients. Continuity evidence is not reused
at a later seam. The dashboard labels low scores as low observed risk and keeps the
temporary in-call profile distinct from real-world identity verification.

The prevention state machine is ephemeral:

```text
monitoring -> held -> verification_pending
                         | verified -> release current evidence epoch
                         | failed   -> blocked/escalated
                         | review   -> remain held
```

Actions and transitions enter the allow-listed metadata audit. No raw speech or
embedding enters the audit. Verification is a clearly labelled simulation.

## Source precedence

1. `STRIVE_architecture_detailed_1 1(1).pdf`: one-page pipeline diagram, visually reviewed.
2. `STRIVE_Architecture_Document(1).md`: detailed mechanics, read in full.
3. `VoiceGuard_PRD_SIH26104.docx`: product journeys, privacy, UI, integration and evaluation goals, read in full. Its architecture was not used.

The official [SIH 2026 problem list](https://sih.gov.in/sih2026PS) could not be independently retrieved in this session. The PS title, sponsor and number are taken from the supplied files, not presented as a freshly verified official listing.

## Runtime flow

Audio arrives as 16 kHz PCM over HTTP/WebSocket or as an in-memory decoded file. A bounded ring buffer produces a 2-second window every second. The feature extractor supplies a CM vector, an independently dimensioned profile vector and time-aligned segment vectors.

Three scoring jobs run in parallel: language-conditioned labelled reference retrieval; similarity to a trusted in-call profile; and local discontinuity across adjacent segments near the new-audio seam. The engine masks unavailable tracks, applies the architecture's age schedule, and smooths the weighted score with alpha 0.7. It then applies separate business-context policy. Audit persists scores and actions, not raw audio or feature vectors.

The implementation supports a browser microphone path. It does not implement SIP/RTP transport, Asterisk, Teams/Zoom interception, or gRPC. Those require additional adapters; REST and WebSocket provide the working integration contract.

## Resolved implementation issues

| Issue in supplied architecture | Implemented resolution |
|---|---|
| Session risk and session similarity share the same name and contradictory inequality. | Expose `session_similarity` separately; define `s_session = clip(1 - similarity, 0, 1)`. Only similarity above 0.70 admits a new window. |
| Last segment of the previous 2-second window and first of the next are one second apart, not adjacent. | Compare adjacent segments around the 1-second seam in the current overlapping window. If there is no nearby valid pair, omit this track. |
| A missing SPS could be treated as risk zero and dilute other evidence. | Mask it and renormalize active weights. Missing global evidence forces uncertainty. |
| Silence or a failed model could produce an apparently safe score. | No public acoustic score without enough unique active audio; errors and silence are explicit reasons. |
| Different vector spaces are combined in the SPS. | Keep separate stores for profile vectors and segment vectors; combine their similarities. Phone IDs restrict matching when available. |
| Bootstrap trust decision is underspecified. | Require all available initial window scores below 0.30, at least three valid windows and four seconds of unique active audio by the 5-second decision. Otherwise permanently block SPS for that call. |
| Unbounded SPS growth. | Keep at most 120 accepted windows per call; clear on end/disconnect/idle expiry. |
| `1 - cosine` ranges from 0 to 2. | Explicitly clip to [0,1]; scores are anomaly indices, not calibrated probabilities. |
| Model language ID described as Indian accent identification. | Report a language code only; hints are labelled operator-supplied. Unknown/small/single-class partitions fall back to global, with an explicit reason. |
| A later risk drop could release a prior alert. | The mock policy latches the hold until a current independent verification is recorded. |

## Preserved settings

| Audio age | Global | Session | Coherence |
|---|---:|---:|---:|
| <15 seconds | 0.80 | 0.00 | 0.20 |
| 15–60 seconds | 0.50 | 0.30 | 0.20 |
| >60 seconds | 0.30 | 0.50 | 0.20 |

Weights are renormalized after unavailable-track masking. Warning is 0.50; alert is 0.75. The UI calls the lower band “Low”, avoiding a claim that identity is verified.

## Hypotheses that remain unproven

- Frozen inference is not universal zero-day detection. No weights are trained here, but the NII encoder was post-trained elsewhere and the reference database still needs labelled data.
- An in-call profile measures consistency with initial audio. It does not prove the caller's real-world identity or ensure the bootstrap is genuine.
- Adjacent natural phonemes can have different embeddings. Boundary discontinuity is a weak, uncalibrated supporting cue and may be confounded by phonetic content or packet loss.
- The self-filtering gate reduces obvious contamination but does not establish resistance to slow poisoning attacks.
- One fixed language label cannot adequately describe all code-mixed calls. The architecture's once-per-call behavior is retained; “mixed” routes to a global pool unless a sufficiently populated explicit partition exists.
- Voice embeddings can be biometric/personal data. They are not declared anonymous merely because raw audio is absent. Clearing arrays is best-effort lifecycle cleanup, not a forensic secure-erasure guarantee.

## Mathematical failure cases retained for investigation

Starting from zero, three maximum-score updates give `1 - 0.6^3 = 0.784`, above the 0.75 alert threshold. This matches the three-chunk alert design goal.

For a mature call with all tracks available, scores `[1,0,0]` fuse to only `0.30`. Thus a completely suspicious global track can be outweighed by stable session/coherence tracks. For blocked SPS with coherence near zero, renormalizing `[0.30,0,0.20]` yields only 0.60 global influence. The tests intentionally preserve these counterexamples. Before deployment, evaluate a calibrated global-evidence floor or revised fusion/policy on held-out data; this build does not quietly change the proposed schedule to hide the issue.

## Operational boundaries

Single process; up to eight calls; one serialized research inference per model instance with parallel feature branches inside it. No horizontal state distribution, model timeout preemption, production IAM or encrypted database implementation. Input ceilings and backpressure are implemented; the prototype needs load tests and process isolation for production. Client capture queues at most three seconds and stops if inference falls behind. No transaction button moves money or contacts third parties.

## Research upgrade

The acoustic profile is 62-dimensional; legacy demo output remains 24-dimensional. CM/profile/CTC extraction uses frozen local models in three parallel jobs. Language ID consumes SpeechBrain posterior confidence; unsupported or low-confidence output becomes `und`. Model hashes and a pipeline fingerprint accompany events.

SPS now keeps persistent profile, phoneme and token-specific FAISS indexes. Accepted windows are added incrementally. Only eviction from the 120-window deque rebuilds indexes; queries are read-only. Cold-start session risk stays `None`, masking unavailable evidence rather than treating it as safe.

The original Markdown is included as `STRIVE_Architecture_Document.md` with corrected model dimensions, checkpoint licensing and EMA semantics. Its remaining throughput and universal Vox-profile claims are marked as source design assumptions.
