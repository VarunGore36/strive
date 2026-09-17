"""Comparable score ablations and measured reports using NumPy/scikit-learn.

Track removals affect score fusion, not the global trust gate used to build SPS.
All main configurations use the same EMA. Raw means are also retained separately.
"""
from itertools import combinations
from typing import Any
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from .engine import scheduled_weights

CONFIGURATIONS = {
    "global_only": (0,), "session_only": (1,), "coherence_only": (2,),
    "track12": (0, 1), "track13": (0, 2), "track23": (1, 2), "strive": (0, 1, 2),
}
LABELS = {"global_only": "Track 1 only", "session_only": "Track 2 only", "coherence_only": "Track 3 only",
          "track12": "Track 1+2", "track13": "Track 1+3", "track23": "Track 2+3", "strive": "STRIVE (full)"}


def ablate_events(events: list[dict], alpha: float = .6, preset: str = "proposed",
                  attack_onset_s: float | None = None) -> dict[str, dict]:
    """Replay score combinations; unavailable evidence never becomes zero risk."""
    result = {}
    for name, selected in CONFIGURATIONS.items():
        previous = None
        trace = []
        for event in events:
            scores = [event["track_scores"][key] for key in ("s_global", "s_session", "s_coherence")]
            weights = np.array(scheduled_weights(event["session_age_s"], preset), dtype=float)
            if len(selected) == 1:
                weights[:] = 1  # A singleton has no relative weighting schedule.
            weights[[i not in selected or scores[i] is None for i in range(3)]] = 0
            if scores[0] is not None and scores[0] >= 0.5:
                session_val = scores[1] if scores[1] is not None else 0.0
                divergence = max(0.0, scores[0] - session_val)
                weights[0] *= (1.0 + 2.0 * divergence)
            raw = None
            # Retain common global-validity admission for all conditional ablations.
            if weights.sum() and scores[0] is not None:
                weights /= weights.sum()
                raw = float(np.dot(weights, [0 if s is None else s for s in scores]))
                previous = alpha * (0 if previous is None else previous) + (1 - alpha) * raw
            eligible = event.get("evidence_eligible", event.get("s_risk") is not None)
            value = previous if eligible and raw is not None else None
            trace.append({"time_s": event["session_age_s"], "score": value,
                          "raw": raw if eligible else None})
        valid = [r for r in trace if r["score"] is not None]
        post = valid if attack_onset_s is None else [r for r in valid if r["time_s"] >= attack_onset_s]
        first_warning = next((r["time_s"] for r in valid if r["score"] >= .5), None)
        first_alert = next((r["time_s"] for r in valid if r["score"] >= .75), None)
        post_warning = next((r["time_s"] for r in post if r["score"] >= .5), None)
        post_alert = next((r["time_s"] for r in post if r["score"] >= .75), None)
        result[name] = {"score": float(np.mean([r["score"] for r in valid])) if valid else None,
            "raw_mean": float(np.mean([r["raw"] for r in valid])) if valid else None,
            "eligible_windows": len(valid), "first_warning_s": first_warning, "first_alert_s": first_alert,
            "post_onset_warning_s": post_warning, "post_onset_alert_s": post_alert,
            "detection_delay_s": post_warning - attack_onset_s if post_warning is not None and attack_onset_s is not None else None,
            "alert_delay_s": post_alert - attack_onset_s if post_alert is not None and attack_onset_s is not None else None,
            "pre_onset_warning": bool(attack_onset_s is not None and first_warning is not None and first_warning < attack_onset_s)}
    return result


def metrics(rows: list[dict], key: str) -> dict[str, Any]:
    """Report ranking/calibration errors, abstentions and empirical operating points."""
    valid = [r for r in rows if r.get(key) is not None]
    out = {"total": len(rows), "scored": len(valid), "abstained": len(rows) - len(valid),
           "coverage": len(valid) / len(rows) if rows else 0.,
           "genuine_scored": sum(r["label"] == 0 for r in valid), "spoof_scored": sum(r["label"] == 1 for r in valid)}
    attacks = [r for r in rows if r["label"] == 1]
    timings = [r.get("timing", {}).get(key, {}) for r in attacks]
    alerts = [t["post_onset_alert_s"] for t in timings if t.get("post_onset_alert_s") is not None]
    delays = [t["detection_delay_s"] for t in timings if t.get("detection_delay_s") is not None]
    out.update(avg_alert_time_s=float(np.mean(alerts)) if alerts else None,
               avg_detection_delay_s=float(np.mean(delays)) if delays else None,
               attacks=len(attacks), attacks_alerted=len(alerts), attacks_without_alert=len(attacks) - len(alerts),
               pre_onset_warnings=sum(bool(t.get("pre_onset_warning")) for t in timings))
    if not valid or len({r["label"] for r in valid}) < 2:
        return {**out, "roc_auc": None, "pr_auc": None, "eer": None, "reason": "Both scored classes are required"}
    y = np.asarray([r["label"] for r in valid]); scores = np.asarray([r[key] for r in valid])
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Metric inputs must be finite [0,1] scores")
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    fnr = 1 - tpr
    crossing = np.where(fpr - fnr >= 0)[0][0]
    if crossing == 0:
        eer = float(fpr[0])
    else:
        a, b = crossing - 1, crossing
        delta = (fpr[b] - fnr[b]) - (fpr[a] - fnr[a])
        fraction = -(fpr[a] - fnr[a]) / delta if delta else 0
        eer = float(fpr[a] + fraction * (fpr[b] - fpr[a]))
    ece = 0.
    bins = np.minimum((scores * 10).astype(int), 9)
    for bucket in range(10):
        mask = bins == bucket
        if mask.any():
            ece += float(mask.mean() * abs(scores[mask].mean() - y[mask].mean()))
    out.update(roc_auc=float(roc_auc_score(y, scores)), pr_auc=float(average_precision_score(y, scores)),
        eer=eer, eer_method="linear interpolation of empirical ROC crossing",
        tpr_at_fpr_1pct=float(tpr[fpr <= .01].max()),
        fpr_resolution=1 / int((y == 0).sum()),
        low_fpr_sample_warning=int((y == 0).sum()) < 100,
        ece_10_bins=ece, brier=float(np.mean((scores - y) ** 2)),
        fpr_at_075=float(np.mean(scores[y == 0] >= .75)), fnr_at_075=float(np.mean(scores[y == 1] < .75)))
    return out


def summarize(records: list[dict]) -> dict:
    """All configs per slice; generator slices share matched real controls."""
    def group(rows: list[dict]) -> dict:
        common = [r for r in rows if all(r.get(name) is not None for name in CONFIGURATIONS)]
        return {"available_cohort": {name: metrics(rows, name) for name in CONFIGURATIONS},
                "common_cohort": {name: metrics(common, name) for name in CONFIGURATIONS}}
    overall = group(records)
    report = {"overall": overall["available_cohort"], "common_cohort": overall["common_cohort"],
              "by_language": {}, "by_codec": {}, "by_generator": {}}
    for field in ("language", "codec"):
        for value in sorted({r[field] for r in records}):
            report["by_" + field][value] = group([r for r in records if r[field] == value])
    for generator in sorted({r["generator"] for r in records if r["label"] == 1}):
        attacks = [r for r in records if r["label"] == 1 and r["generator"] == generator]
        strata = {(r["language"], r["codec"]) for r in attacks}
        controls = [r for r in records if r["label"] == 0 and (r["language"], r["codec"]) in strata]
        report["by_generator"][generator] = {**group(attacks + controls),
            "control_policy": "All genuine clips matching attack language/codec strata; reused across generators"}
    return report


def markdown_table(summary: dict[str, dict], title: str | None = None) -> str:
    """Format only measured values; absence is displayed as N/A."""
    lines = [title, ""] if title else []
    lines += ["| Configuration | EER (%) | AUC | TPR@1%FPR | Brier | Avg Alert Time (s) | Scored/total |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    def number(value: float | None, digits: int = 3, scale: float = 1) -> str:
        return "N/A" if value is None else f"{value * scale:.{digits}f}"
    for name in CONFIGURATIONS:
        m = summary[name]
        values = [LABELS[name], number(m.get("eer"), 2, 100), number(m.get("roc_auc")),
                  number(m.get("tpr_at_fpr_1pct")), number(m.get("brier")), number(m.get("avg_alert_time_s"), 1),
                  f"{m['scored']}/{m['total']}"]
        if name == "strive":
            values = ["**" + v + "**" for v in values]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
