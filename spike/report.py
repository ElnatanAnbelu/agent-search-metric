#!/usr/bin/env python3
"""Join the recovered search behaviour to how each run actually scored.

The question the spike exists to answer is whether "how many distinct configurations did
this agent evaluate" is recoverable and whether it relates to the outcome. It is a sample,
not a result: with a dozen runs nothing here is significant, and the point is to find out
whether the larger analysis is worth doing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import DATA, analyse  # noqa: E402

# Each scenario is scored on its own primary metric, and lower is better for the latency
# ones. Comparing across scenarios is meaningless, so runs are ranked within a scenario.
PRIMARY = {
    "A": ("ttft", "p50", False),
    "B": ("tpot", "p50", False),
    "C": ("request_throughput_req_per_s", None, True),
    "D": ("tpot", "p50", False),
}


def outcome(run_dir: Path, scenario: str) -> tuple[bool, float | None]:
    """(passed the quality gate, primary metric) for one run."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    passed = bool((metrics.get("quality_check") or {}).get("pass"))
    key, stat, _ = PRIMARY[scenario]
    values = []
    for profile in (metrics.get("profiles") or {}).values():
        v = profile.get(key)
        if isinstance(v, dict):
            v = v.get(stat)
        if isinstance(v, (int, float)) and v > 0:
            values.append(float(v))
    return passed, (sum(values) / len(values) if values else None)


def main() -> int:
    metas = {r["anon_id"]: r for r in json.loads((DATA / "sample.json").read_text())}
    rows = []
    for run_id in sorted(metas):
        d = DATA / "runs" / run_id
        if not (d / "trace.jsonl").exists():
            continue
        a = analyse(d, metas[run_id])
        meta = json.loads((d / "run_meta.json").read_text())
        passed, primary = outcome(d, a.scenario)
        rows.append({
            "run": a.run, "harness": a.harness, "agent": a.agent, "scenario": a.scenario,
            "distinct": a.distinct, "launches": len(a.launched), "measures": a.measures,
            "passed": passed, "primary": primary,
            "hacked": bool(meta.get("invalid_or_reward_hack")),
        })

    print(f"{'run':10} {'agent':22} {'sc':3} {'distinct':>8} {'relaunch':>8} {'measure':>7} "
          f"{'gate':>5}  primary")
    for r in rows:
        relaunch = r["launches"] - r["distinct"]
        primary = f"{r['primary']:.4f}" if r["primary"] else "-"
        print(f"{r['run']:10} {r['agent']:22} {r['scenario']:3} {r['distinct']:8} "
              f"{relaunch:8} {r['measures']:7} {'pass' if r['passed'] else 'FAIL':>5}  {primary}")

    passed = [r for r in rows if r["passed"]]
    failed = [r for r in rows if not r["passed"]]
    print(f"\n{len(passed)}/{len(rows)} runs passed the quality gate.")
    for label, group in (("passed", passed), ("failed", failed)):
        if group:
            ds = sorted(r["distinct"] for r in group)
            print(f"  {label:6} distinct configurations: {ds}  median {ds[len(ds) // 2]}")

    # Within a scenario, does trying more configurations track a better primary metric?
    print("\nwithin-scenario ordering (best primary metric first):")
    for scenario in sorted({r["scenario"] for r in rows}):
        group = [r for r in rows if r["scenario"] == scenario and r["passed"] and r["primary"]]
        if len(group) < 2:
            print(f"  {scenario}: {len(group)} scored run(s), nothing to order")
            continue
        _, _, higher_better = PRIMARY[scenario]
        group.sort(key=lambda r: r["primary"], reverse=higher_better)
        print(f"  {scenario}: " + "  ".join(f"{r['run']}(cfg={r['distinct']})" for r in group))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
