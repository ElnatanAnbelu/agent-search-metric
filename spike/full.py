#!/usr/bin/env python3
"""Run both metrics over every downloaded trajectory and report by agent and harness.

The spike answered "is this recoverable at all" on twelve runs. This asks the questions
that need the whole set: how often agents ship what they never measured, whether that
varies by model or harness, and how much of the answer is limited by what the traces
actually recorded rather than by anything the agents did.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import DATA, analyse as analyse_search  # noqa: E402
from submission import TIE, analyse as analyse_submission  # noqa: E402

VERDICTS = ("best", "tied", "worse", "unmeasured", "unjudged")


def rate(part: int, whole: int) -> str:
    return f"{part}/{whole}" + (f" ({part / whole:.0%})" if whole else "")


def main() -> int:
    manifest = json.loads((DATA / "manifest.json").read_text())
    metas = {r["anon_id"]: r for r in manifest["runs"]}
    present = sorted(p.name for p in (DATA / "runs").iterdir()
                     if (p / "trace.jsonl").exists() and p.name in metas)
    print(f"{len(present)} of {manifest['n_runs']} runs present\n")

    rows = []
    for run_id in present:
        meta = metas[run_id]
        d = DATA / "runs" / run_id
        search = analyse_search(d, meta)
        sub = analyse_submission(d, meta)
        # A run in flight, or one the dataset ships without its metadata, is skipped
        # rather than crashing a 269-run pass.
        meta_file = d / "run_meta.json"
        run_meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        rows.append({
            "run": run_id, "harness": meta["harness"], "agent": meta["agent"],
            "scenario": meta["scenario"],
            "distinct": search.distinct, "launches": len(search.launched),
            "measured": len(sub.measured),
            "configs_measured": len({c for c, _ in sub.measured}),
            "verdict": sub.verdict, "verdict_full": sub.verdict_full, "regret": sub.regret,
            "full_readings": len(sub.full_only),
            "hacked": bool(run_meta.get("invalid_or_reward_hack")),
        })

    json.dump(rows, (DATA / "analysis.json").open("w"), indent=1)

    counts = Counter(r["verdict"] for r in rows)
    judged = [r for r in rows if r["verdict"] != "unjudged"]
    print("== Did the agent submit the best configuration it measured?")
    for v in VERDICTS:
        print(f"   {v:11} {counts[v]:4}")
    if judged:
        n = len(judged)
        counts_j = Counter(r["verdict"] for r in judged)
        print(f"\n   of {n} judgeable runs:")
        print(f"     submitted their best            {rate(counts_j['best'], n)}")
        print(f"     submitted a tie (within {TIE:.0%})      {rate(counts_j['tied'], n)}")
        print(f"     submitted a worse measured one  {rate(counts_j['worse'], n)}")
        print(f"     submitted one never measured    {rate(counts_j['unmeasured'], n)}")
        regrets = [r["regret"] for r in judged if r["regret"] is not None]
        if regrets:
            print(f"     median regret when worse    {statistics.median(regrets):+.1%}")

    strict = [r for r in rows if r["verdict_full"] != "unjudged"]
    if strict:
        cs = Counter(r["verdict_full"] for r in strict)
        print(f"\n   Same question over provably full-scale readings only ({len(strict)} runs):")
        for v in ("best", "tied", "worse", "unmeasured"):
            print(f"     {v:12} {rate(cs[v], len(strict))}")

    print("\n== Search breadth (distinct configurations launched)")
    ds = sorted(r["distinct"] for r in rows)
    print(f"   median {statistics.median(ds):.0f}   mean {statistics.mean(ds):.1f}   "
          f"max {max(ds)}   zero-recovered {sum(1 for d in ds if d == 0)}")
    relaunch = [r["launches"] - r["distinct"] for r in rows]
    print(f"   relaunches of an already-tried configuration: median "
          f"{statistics.median(relaunch):.0f}, max {max(relaunch)}")

    print("\n== By harness")
    print(f"   {'harness':10} {'runs':>5} {'med cfg':>8} {'judged':>7} {'never measured':>15}")
    by_harness = defaultdict(list)
    for r in rows:
        by_harness[r["harness"]].append(r)
    for harness, group in sorted(by_harness.items()):
        j = [r for r in group if r["verdict"] != "unjudged"]
        n = sum(1 for r in j if r["verdict"] == "unmeasured")
        print(f"   {harness:10} {len(group):5} "
              f"{statistics.median([r['distinct'] for r in group]):8.0f} "
              f"{len(j):7} {rate(n, len(j)):>15}")

    print("\n== By agent (models with at least 6 runs)")
    print(f"   {'agent':24} {'runs':>5} {'med cfg':>8} {'judged':>7} {'never measured':>15}")
    by_agent = defaultdict(list)
    for r in rows:
        by_agent[r["agent"]].append(r)
    for agent, group in sorted(by_agent.items(),
                               key=lambda kv: -statistics.median([r["distinct"] for r in kv[1]])):
        if len(group) < 6:
            continue
        j = [r for r in group if r["verdict"] != "unjudged"]
        n = sum(1 for r in j if r["verdict"] == "unmeasured")
        print(f"   {agent:24} {len(group):5} "
              f"{statistics.median([r['distinct'] for r in group]):8.0f} "
              f"{len(j):7} {rate(n, len(j)):>15}")

    print("\n== What limits the answer")
    print(f"   runs with no recoverable configuration  {rate(sum(1 for r in rows if not r['distinct']), len(rows))}")
    print(f"   runs with no recoverable measurement    {rate(sum(1 for r in rows if not r['measured']), len(rows))}")
    print(f"   runs measuring only one configuration   {rate(sum(1 for r in rows if r['configs_measured'] == 1), len(rows))}")
    print(f"   flagged invalid or reward-hacked        {rate(sum(1 for r in rows if r['hacked']), len(rows))}")
    print("\n   per-run detail written to data/analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
