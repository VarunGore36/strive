"""Draw a balanced, speaker-aware sample from a large official manifest.

ASVspoof 2019 LA is heavily imbalanced (roughly 1 bonafide to 9 spoof) and far
larger than a smoke benchmark needs. Sampling matters for two reasons:

1. Low-FPR operating points need genuine clips. The finest resolvable false
   positive rate is 1/n0, so quoting TPR at 1% FPR requires n0 >= 100 and is
   only stable near n0 >= 1000. The earlier 300-clip run had n0 = 65, which is
   why its TPR@1%FPR was flagged unquotable.

2. Spoof clips must be spread across attack generators, or the result describes
   whichever attack happened to be sampled rather than the condition as a whole.

Speaker identity is preserved so downstream leakage checks still work.
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def sample(rows: list[dict], per_class: int, seed: int) -> list[dict]:
    """Balanced draw: per_class bonafide, per_class spoof spread over generators."""
    rng = random.Random(seed)
    bonafide = [r for r in rows if r["label"] == "0"]
    by_generator = defaultdict(list)
    for r in rows:
        if r["label"] == "1":
            by_generator[r["generator"]].append(r)

    rng.shuffle(bonafide)
    picked = bonafide[:per_class]
    if len(picked) < per_class:
        raise ValueError(f"Only {len(picked)} bonafide rows available, wanted {per_class}")

    generators = sorted(by_generator)
    if not generators:
        raise ValueError("No spoof rows in manifest")
    # Even quota per generator, then top up from the pooled remainder so the
    # total is exact even when generators have unequal counts.
    quota = per_class // len(generators) + 1
    spoof, leftovers = [], []
    for g in generators:
        clips = by_generator[g]
        rng.shuffle(clips)
        spoof.extend(clips[:quota])
        leftovers.extend(clips[quota:])
    rng.shuffle(leftovers)
    spoof = spoof[:per_class] if len(spoof) >= per_class else spoof + leftovers[:per_class - len(spoof)]
    rng.shuffle(spoof)
    return picked + spoof


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("--output", required=True)
    p.add_argument("--per-class", type=int, default=1500)
    p.add_argument("--seed", type=int, default=26104)
    args = p.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    picked = sample(rows, args.per_class, args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(picked)

    generators = defaultdict(int)
    for r in picked:
        if r["label"] == "1":
            generators[r["generator"]] += 1
    print(f"Wrote {len(picked)} rows to {out}")
    print(f"  bonafide {sum(r['label'] == '0' for r in picked)}, "
          f"spoof {sum(r['label'] == '1' for r in picked)}")
    print(f"  speakers {len({r['speaker_id'] for r in picked})}")
    print(f"  generators {dict(sorted(generators.items()))}")


if __name__ == "__main__":
    main()
