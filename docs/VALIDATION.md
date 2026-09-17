> Original MVP validation record. Current upgrade results are in `RESEARCH_UPGRADE.md` and `evidence/research-upgrade-tests.xml`. The demo benchmark/replay has been regenerated for version 0.2.0.

# STRIVE MVP validation report

**Result: the implemented streaming mechanics and mock workflow run successfully. The full pretrained voice-cloning detection architecture is not yet validated.**

The code follows the STRIVE PDF/Markdown three-track architecture. The DOCX was used for MVP behavior only. See `docs/ARCHITECTURE.md` for corrections and `docs/REQUIREMENTS.md` for unmet requirements.

## Executed checks

- 32 Python tests passed: ring-buffer overlap, sample-rate conversion, score polarity, bootstrap gates, missing-track masking, EMA, call isolation, profile erasure, sequence validation, REST/WebSocket flows, input validation, authentication/origin checks, privacy audit, mock hold/verification, and evaluation leakage handling.
- JavaScript syntax checks passed for dashboard and AudioWorklet.
- AudioWorklet framing checks passed at 16,000, 44,100 and 48,000 Hz input rates; each produced exactly two 16,000-sample frames from two seconds of audio.
- The four scenarios ran through the real FastAPI routes, not hard-coded UI score curves.
- An actual local Uvicorn HTTP smoke test checked health, 39 streaming windows, transaction hold, mock verification, release and call teardown. See `evidence/http-smoke.json`.
- Two dependency deprecation warnings occurred in the test client; no test failures were reported. They do not establish research dependency compatibility.

## Measured procedural scenarios

All source audio here is generated engineering test signals. Family 0 is a harmonic reference signal; family 1 has a deliberately different high-frequency/noisy spectrum. Neither label is a recording of a genuine human or a cloned voice. The reference database has 48 procedural embeddings, not the architecture's proposed multilingual corpus.

Each run contains 40 seconds of audio, giving 39 windows. Business context is zero for this benchmark. Audio timestamps below are seconds into the input, not wall-clock execution time.

| Scenario | Bootstrap | Stored windows | First warning | First alert | Final risk | p95 compute |
|---|---|---:|---|---|---:|---:|
| steady | trusted | 39 | No warning | No alert | 0.000 | 13.22 ms |
| switch | trusted | 17 | 22 s | 36 s | 0.806 | 11.41 ms |
| suspicious_start | blocked | 0 | 4 s | 8 s | 0.729 | 8.52 ms |
| silence | blocked | 0 | No warning | No alert | Unavailable | 0.08 ms |

The measurements describe a single local CPU run of the DSP surrogate and a tiny FAISS corpus. They exclude neural-model inference, large-index retrieval, network transport and browser capture. **They cannot substantiate a 200 ms T4 or production real-time claim.**

## Architecture findings

1. **The mid-call change is detected slowly under the proposed fusion.** The test switches at 18 seconds. A warning occurs at 22 seconds and an alert at 36 seconds: four and eighteen seconds after the change. Treat this as a weakness to investigate with real speech, not as an acceptable detection target.
2. **The session gate prevents obvious fixture contamination.** In the switch run, the profile stops at 17 accepted windows. In the suspicious-start run it stays empty. This does not prove resistance to adversarial slow poisoning.
3. **The score can decline despite suspicious global evidence.** The suspicious-start fixture alerts at eight seconds, then ends below 0.75 as age weights change. The mock workflow latches the earlier hold; the acoustic schedule itself is unchanged.
4. **The mature schedule can suppress global evidence.** With all tracks available, `[1, 0, 0]` at age >60 seconds gives a raw score of 0.30. A static clone that resembles the initial profile may therefore challenge the proposed fusion.
5. **EMA warm-up fixed.** Three maximum-score updates from zero now yield `1 - 0.6^3 = 0.784`, above the 0.75 alert threshold, matching the three-chunk alert design goal.
6. **Coherence needs empirical validation.** The overlap indexing bug is corrected, but similarity between different adjacent phones is not inherently a genuine-vs-synthetic discriminator.

## Unexecuted or incomplete

- NII pretrained checkpoint export/inference and XLSR pretrained phoneme inference.
- SpeechBrain automatic language recognition on real speech.
- Exact Vox-Profile 283-dimensional branch: the source interface was not specified or verified; the research default is a declared acoustic-profile substitute.
- Genuine-vs-cloned speech accuracy, calibrated risk, EER/ROC-AUC, unseen-generator robustness, telephony robustness and Indian-language/code-mixed performance. Evaluation scripts exist; no such metrics are fabricated.
- Physical microphone permission/capture, browser visual testing, and a live Hindi/Hinglish demonstration.
- Windows installation, Docker build and GPU execution.
- SIP/RTP/Asterisk, gRPC, external banking, SMS/webhook integration and real authentication/verification providers.

The official SIH page could not be independently retrieved. Problem-statement context comes from the three supplied documents; source file hashes are in `evidence/source-manifest.json`.

## Next experiment that answers the user's main question

Install/export the actual frozen models; resolve the Vox-Profile short-window contract; build a licensed, consented real/spoof reference corpus; collect speaker/source-disjoint test calls with known injection timestamps; compare Track 1 alone against the complete three-track system. Measure false positives, missed attacks, per-language/codecs, and time-to-alert on the intended hardware. Only that experiment can establish whether STRIVE improves voice-cloning detection, rather than merely demonstrating that its components can exchange scores.
