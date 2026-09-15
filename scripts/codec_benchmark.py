"""CODEC-04: smoke benchmark of detection and latency across codec conditions.

Builds a reference index from CLEAN audio of one speaker group, then scores a
speaker-disjoint evaluation group under every codec condition. Reports score
distributions, EER/ROC-AUC where the cohort permits, latency and failure counts.

This runs with whatever extractor is configured. With the default DSP surrogate
the scores are NOT a deepfake verdict -- the point is then to measure the harness
and the channel, and to establish the floor a real model has to beat. The report
records which extractor produced it so the two can never be confused.

    python scripts/codec_benchmark.py --matrix data/manifests/<set>/matrix.csv \\
           --output evidence/codec-benchmark.json
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from strive.ablation import metrics
from strive.audio import RATE, decode
from strive.config import Settings
from strive.engine import Call
from strive.features import DSPExtractor
from strive.retrieval import ReferenceIndex
from strive.telemetry import percentiles


def load_matrix(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Matrix manifest is empty")
    for row in rows:
        if row["label"] not in ("0", "1"):
            raise ValueError("Labels must be 0 (bonafide) or 1 (spoof)")
    return rows


def split_speakers(rows: list[dict], reference_fraction: float, seed: int) -> tuple[set, set]:
    """Speaker-disjoint split. A speaker never appears on both sides."""
    speakers = sorted({r["speaker_id"] for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)
    cut = max(1, int(len(speakers) * reference_fraction))
    reference, evaluation = set(speakers[:cut]), set(speakers[cut:])
    if not evaluation:
        raise ValueError("Speaker split left no evaluation speakers; lower --reference-fraction")
    return reference, evaluation


def read_audio(path: str) -> np.ndarray:
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != RATE:
        return decode(Path(path).read_bytes())
    return np.clip(data.mean(axis=1), -1, 1).astype(np.float32)


def build_index(rows: list[dict], extractor) -> ReferenceIndex:
    """Reference index from CLEAN audio only, so the index is not codec-biased."""
    vectors, labels, languages, ids = [], [], [], []
    for row in rows:
        audio = read_audio(row["path"])
        window = round(2 * RATE)
        if len(audio) < window:
            continue
        # One centre window per clip keeps the index balanced across clips.
        start = (len(audio) - window) // 2
        vectors.append(extractor.extract(audio[start:start + window]).cm)
        labels.append(int(row["label"]))
        languages.append(row["language"])
        ids.append(row["source_id"])
    if len({*labels}) < 2:
        raise ValueError("Reference index needs both bonafide and spoof clips")
    return ReferenceIndex(vectors, labels, languages,
                          {"extractor_id": extractor.id, "demo_only": extractor.is_surrogate,
                           "description": f"CODEC-04 reference index, {len(vectors)} clean clips"}, ids)


def score_clip(row: dict, settings: Settings, extractor, index) -> dict:
    """Stream one clip through the real engine and summarize its risk."""
    call = Call(settings, extractor, index, language=row["language"], source="codec_benchmark")
    events, failures = [], 0
    started = time.perf_counter()
    try:
        audio = read_audio(row["path"])
        for sequence, offset in enumerate(range(0, len(audio), RATE)):
            frame = audio[offset:offset + RATE]
            if len(frame):
                events.extend(call.feed(frame, sequence))
    except Exception as error:
        failures = 1
        return {"source_id": row["source_id"], "label": int(row["label"]), "condition": row["condition"],
                "speaker_id": row["speaker_id"], "generator": row["generator"], "score": None,
                "failed": True, "error": f"{type(error).__name__}: {error}", "windows": len(events),
                "wall_ms": (time.perf_counter() - started) * 1000}
    finally:
        call.close()

    eligible = [e for e in events if e["s_risk"] is not None]
    latency = [e["latency_ms"]["end_to_end"] for e in events]
    channel = [e["channel"] for e in events if e["channel"]["quality"] is not None]
    # `s_risk` is gated by min_voiced_s, a LIVE-CALL safety rule: it refuses to
    # publish a risk until enough voiced audio has accumulated. ASVspoof clips are
    # utterances (median ~3 s), so most never clear that gate. Lowering the gate to
    # make numbers appear would be tuning for the benchmark, so instead the primary
    # metric here is the detection track itself, which is produced per window.
    track = [e["track_scores"]["s_global"] for e in events
             if e["track_scores"]["s_global"] is not None]
    return {"source_id": row["source_id"], "label": int(row["label"]), "condition": row["condition"],
            "speaker_id": row["speaker_id"], "generator": row["generator"],
            "score": float(np.mean(track)) if track else None,
            "gated_risk": eligible[-1]["s_risk"] if eligible else None,
            "passed_evidence_gate": bool(eligible),
            "failed": False, "windows": len(events), "eligible_windows": len(eligible),
            "latency_ms": latency,
            "channel_quality": float(np.mean([c["quality"] for c in channel])) if channel else None,
            "bandwidth_hz": float(np.mean([c["estimated_bandwidth_hz"] for c in channel
                                           if c["estimated_bandwidth_hz"] is not None])) if channel else None,
            "snr_db": float(np.mean([c["snr_db"] for c in channel if c["snr_db"] is not None])) if channel else None,
            "wall_ms": (time.perf_counter() - started) * 1000}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", required=True, help="paired manifest from augment.py matrix")
    parser.add_argument("--output", default="evidence/codec-benchmark.json")
    parser.add_argument("--markdown", default="evidence/codec-benchmark.md")
    parser.add_argument("--reference-manifest",
                        help="clean manifest for the index; when given, the "
                             "matrix is evaluated in full and no speaker split "
                             "is performed (the official splits are already "
                             "speaker- and generator-disjoint)")
    parser.add_argument("--reference-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=26104)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--hop", type=float, default=1.0)
    parser.add_argument("--limit", type=int, help="evaluate only the first N clips per condition")
    args = parser.parse_args()

    rows = load_matrix(Path(args.matrix))
    conditions = sorted({r["condition"] for r in rows})

    if args.reference_manifest:
        # Official splits: reference and test are already speaker-disjoint, and
        # in ASVspoof 2019 LA their attack generators are disjoint too (A01-A06
        # vs A07-A19), so this is an UNSEEN-GENERATOR evaluation.
        reference_rows = load_matrix(Path(args.reference_manifest))
        reference_speakers = {r["speaker_id"] for r in reference_rows}
        eval_speakers = {r["speaker_id"] for r in rows}
        overlap = reference_speakers & eval_speakers
        if overlap:
            raise SystemExit(f"Speaker leakage: {len(overlap)} shared speakers")
        ref_gen = {r["generator"] for r in reference_rows if r["label"] == "1"}
        eval_gen = {r["generator"] for r in rows if r["label"] == "1"}
        print(f"reference generators {sorted(ref_gen)}")
        print(f"evaluation generators {sorted(eval_gen)}")
        print(f"unseen-generator: {'YES' if not (ref_gen & eval_gen) else 'NO'}")
    else:
        reference_speakers, eval_speakers = split_speakers(rows, args.reference_fraction, args.seed)
        clean = [r for r in rows if r["condition"] == "clean"]
        if not clean:
            raise SystemExit("Matrix has no 'clean' condition to build the index from")
        reference_rows = [r for r in clean if r["speaker_id"] in reference_speakers]

    extractor = DSPExtractor()
    settings = Settings(window_s=args.window, stride_s=args.hop, audit_path=":memory:")
    print(f"Conditions: {', '.join(conditions)}")
    print(f"Speakers: {len(reference_speakers)} reference / {len(eval_speakers)} evaluation "
          f"(disjoint)")
    print(f"Building reference index from {len(reference_rows)} clean clips...")
    index = build_index(reference_rows, extractor)
    print(f"  {len(index.x)} vectors, extractor={extractor.id}, surrogate={extractor.is_surrogate}")

    records, per_condition = [], defaultdict(list)
    for condition in conditions:
        selected = [r for r in rows if r["condition"] == condition
                    and (args.reference_manifest or r["speaker_id"] in eval_speakers)]
        if args.limit:
            selected = selected[:args.limit]
        print(f"\n{condition}: scoring {len(selected)} clips", flush=True)
        for done, row in enumerate(selected, 1):
            result = score_clip(row, settings, extractor, index)
            records.append(result)
            per_condition[condition].append(result)
            if done % 25 == 0:
                print(f"  {done}/{len(selected)}", flush=True)

    report = {
        "run_id": f"codec04-{int(time.time())}",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor() or None,
                     "python": platform.python_version()},
        "extractor": {"id": extractor.id, "is_surrogate": extractor.is_surrogate},
        "reference_index": {"vectors": len(index.x), "clean_clips": len(reference_rows),
                            "demo_only": index.metadata.get("demo_only")},
        "split": {"reference_speakers": len(reference_speakers),
                  "evaluation_speakers": len(eval_speakers), "speaker_disjoint": True,
                  "reference_generators": sorted({r["generator"] for r in reference_rows
                                                  if r["label"] == "1"}),
                  "evaluation_generators": sorted({r["generator"] for r in rows
                                                   if r["label"] == "1"}),
                  "unseen_generator": not ({r["generator"] for r in reference_rows if r["label"] == "1"}
                                           & {r["generator"] for r in rows if r["label"] == "1"}),
                  "seed": args.seed},
        "geometry": {"window_s": args.window, "hop_s": args.hop},
        "conditions": {},
        "score_definition": (
            "Per-clip score is the mean s_global (labelled-reference retrieval track) over "
            "windows where it was available. The gated s_risk is reported separately as "
            "passed_live_call_evidence_gate: min_voiced_s=4.0 s is a live-call safety rule and "
            "most ASVspoof utterances are shorter, so it is a product-coverage fact here, not a "
            "detection metric. The gate was NOT lowered to produce results."),
        "interpretation": (
            "Scores come from the extractor named above. A surrogate extractor makes these "
            "harness and channel measurements, NOT a deepfake detection result. EER/AUC near "
            "0.5 means no separation, which is the expected floor for DSP features and the "
            "baseline a trained countermeasure must beat."),
    }

    for condition in conditions:
        results = per_condition[condition]
        scored = [r for r in results if r["score"] is not None]
        latency = [ms for r in results for ms in r.get("latency_ms", [])]
        report["conditions"][condition] = {
            "clips": len(results),
            "failures": sum(r["failed"] for r in results),
            "scored": len(scored),
            "abstained": len(results) - len(scored),
            "passed_live_call_evidence_gate": sum(bool(r.get("passed_evidence_gate")) for r in results),
            "metrics": metrics([{k: v for k, v in r.items()} for r in results], "score"),
            "score_distribution": {
                "bonafide": percentiles([r["score"] for r in scored if r["label"] == 0], (25, 50, 75)),
                "spoof": percentiles([r["score"] for r in scored if r["label"] == 1], (25, 50, 75))},
            "latency_ms": percentiles(latency, (50, 95)),
            "channel": {
                "quality": percentiles([r["channel_quality"] for r in results
                                        if r.get("channel_quality") is not None], (50,)),
                "bandwidth_hz": percentiles([r["bandwidth_hz"] for r in results
                                             if r.get("bandwidth_hz") is not None], (50,)),
                "snr_db": percentiles([r["snr_db"] for r in results
                                       if r.get("snr_db") is not None], (50,))},
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))

    lines = [f"# CODEC-04 smoke benchmark", "",
             f"Run `{report['run_id']}` — extractor `{extractor.id}` "
             f"(surrogate: {extractor.is_surrogate})", "",
             f"{len(reference_speakers)} reference / {len(eval_speakers)} evaluation speakers, "
             f"disjoint. Reference index: {len(index.x)} clean clips.", "",
             "| Condition | Clips | Scored | Fail | EER % | AUC | Gate pass | Bandwidth Hz | SNR dB | Quality | p50 ms | p95 ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def cell(value, digits=3, scale=1.0):
        return "N/A" if value is None else f"{value * scale:.{digits}f}"

    for condition in conditions:
        c = report["conditions"][condition]
        m, ch, lat = c["metrics"], c["channel"], c["latency_ms"]
        lines.append(f"| `{condition}` | {c['clips']} | {c['scored']} | {c['failures']} | "
                     f"{cell(m.get('eer'), 2, 100)} | {cell(m.get('roc_auc'))} | "
                     f"{c['passed_live_call_evidence_gate']} | "
                     f"{cell(ch['bandwidth_hz']['p50'], 0)} | {cell(ch['snr_db']['p50'], 1)} | "
                     f"{cell(ch['quality']['p50'])} | {cell(lat['p50'], 2)} | {cell(lat['p95'], 2)} |")
    lines += ["", "> " + report["interpretation"]]
    Path(args.markdown).write_text("\n".join(lines) + "\n")

    print(f"\nWrote {output} and {args.markdown}")
    print("\n".join(lines[5:]))


if __name__ == "__main__":
    main()
