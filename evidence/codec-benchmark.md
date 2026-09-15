# CODEC-04 smoke benchmark

Run `codec04-1788870525` — extractor `dsp-surrogate-v1` (surrogate: True)

31 reference / 31 evaluation speakers, disjoint. Reference index: 129 clean clips.

| Condition | Clips | Scored | Fail | EER % | AUC | Gate pass | Bandwidth Hz | SNR dB | Quality | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 152 | 122 | 0 | 37.01 | 0.707 | 5 | 3625 | 44.7 | 0.749 | 5.61 | 6.27 |
| `g711_ulaw` | 152 | 122 | 0 | 37.49 | 0.676 | 4 | 2672 | 44.0 | 0.708 | 5.47 | 6.18 |
| `narrowband_8k` | 152 | 122 | 0 | 37.40 | 0.676 | 4 | 2672 | 44.7 | 0.708 | 5.48 | 6.24 |
| `noise_10db` | 152 | 122 | 0 | 42.14 | 0.639 | 28 | 7062 | 19.6 | 0.898 | 6.24 | 6.50 |
| `opus_16k` | 152 | 122 | 0 | 37.43 | 0.688 | 4 | 3094 | 44.9 | 0.730 | 5.50 | 6.17 |
| `opus_32k` | 152 | 122 | 0 | 39.66 | 0.686 | 4 | 3135 | 45.2 | 0.727 | 5.52 | 6.25 |

> Scores come from the extractor named above. A surrogate extractor makes these harness and channel measurements, NOT a deepfake detection result. EER/AUC near 0.5 means no separation, which is the expected floor for DSP features and the baseline a trained countermeasure must beat.
