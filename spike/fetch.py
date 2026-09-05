#!/usr/bin/env python3
"""Download a sample of InferenceBench trajectories.

The dataset is Apache-2.0 and public, so this is a plain HTTP fetch with no token and
no huggingface library. Files land under data/runs/<run>/ and are never re-downloaded.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/aisa-group/InferenceBench-Trajectories/resolve/main"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FILES = ("trace.jsonl", "metrics.json", "run_meta.json")


def get(path: str, dest: Path) -> bool:
    """Fetch one file. Returns False when the dataset does not have it.

    Not every run ships every file, and one 404 must not end a 269-run download.
    """
    if dest.exists() and dest.stat().st_size:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(f"{BASE}/{path}", timeout=120) as r, dest.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except urllib.error.HTTPError as e:
        dest.unlink(missing_ok=True)
        if e.code == 404:
            return False
        raise
    return True


def manifest() -> dict:
    dest = DATA / "manifest.json"
    get("manifest.json", dest)
    return json.loads(dest.read_text())


def sample(runs: list[dict], per_harness: int) -> list[dict]:
    """Spread the sample across harnesses and scenarios rather than taking a prefix.

    Run ids are ordered by harness in places, so the first N runs are not a sample of
    anything. Taking per_harness from each harness, cycling scenarios, keeps the spike
    honest about coverage.
    """
    picked: list[dict] = []
    for harness in sorted({r["harness"] for r in runs}):
        pool = [r for r in runs if r["harness"] == harness]
        by_scenario: dict[str, list[dict]] = {}
        for r in pool:
            by_scenario.setdefault(r["scenario"], []).append(r)
        order = sorted(by_scenario)
        i = 0
        while len([p for p in picked if p["harness"] == harness]) < per_harness and order:
            key = order[i % len(order)]
            if by_scenario[key]:
                picked.append(by_scenario[key].pop(0))
            elif all(not v for v in by_scenario.values()):
                break
            i += 1
    return picked


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else "4"
    m = manifest()
    chosen = m["runs"] if arg == "all" else sample(m["runs"], int(arg))
    print(f"{m['n_runs']} runs available, fetching {len(chosen)}")
    missing = 0
    for n, r in enumerate(chosen, 1):
        for name in FILES:
            if not get(f"runs/{r['anon_id']}/{name}", DATA / "runs" / r["anon_id"] / name):
                missing += 1
                print(f"  {r['anon_id']}: no {name}")
        if n % 25 == 0 or n == len(chosen):
            print(f"  {n}/{len(chosen)}")
    if missing:
        print(f"{missing} file(s) absent from the dataset")
    (DATA / "sample.json").write_text(json.dumps(chosen, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
