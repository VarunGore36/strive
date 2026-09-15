"""Resume the ASVspoof 2019 LA download from a range-capable mirror.

Zenodo serves the archive but has been returning HTTP 504 partway through, and
its retries eventually exhaust. Edinburgh DataShare hosts the same file and
advertises `Accept-Ranges: bytes`, so an interrupted transfer can be continued
rather than restarted.

Any existing `LA.zip.partial` is reused. The published md5 is verified before the
file is committed, so mixing bytes from two mirrors is safe: a mismatch fails
loudly instead of producing a corrupt archive.

    python scripts/resume_asv.py --root /path/to/asv2019
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_data import download, extract_archive, asv_rows, write_manifests

# Edinburgh DataShare bitstream for LA.zip. Verified range-capable (HTTP 206).
EDINBURGH = ("https://datashare.ed.ac.uk/server/api/core/bitstreams/"
             "a9f87c35-f055-4015-80e2-2fdff0d46269/content")
# Published on the Zenodo record for the same artifact.
LA_MD5 = "md5:30c98f11d8b2bc21f2c257bfd78bb5c5"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--url", default=EDINBURGH)
    p.add_argument("--checksum", default=LA_MD5)
    p.add_argument("--attempts", type=int, default=40,
                   help="the link is slow and flaky; many short attempts beat few long ones")
    p.add_argument("--output", default="data/manifests/asv2019")
    p.add_argument("--no-extract", action="store_true")
    args = p.parse_args()

    root = Path(args.root).resolve()
    target = root / "archives" / "LA.zip"
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        print(f"Resuming from {partial.stat().st_size / 1e9:.2f} GB already on disk")
    if target.exists():
        print(f"{target} already present; skipping download")
    else:
        download(args.url, target, args.checksum, attempts=args.attempts)
        print(f"Verified and committed: {target} ({target.stat().st_size / 1e9:.2f} GB)")

    if args.no_extract:
        return
    print("Extracting...")
    extract_archive(target, root)
    rows = asv_rows(root, "asv2019")
    import json
    print(json.dumps(write_manifests(rows, Path(args.output)), indent=2))


if __name__ == "__main__":
    main()
