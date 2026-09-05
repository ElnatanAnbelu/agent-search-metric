#!/usr/bin/env python3
"""Check the classifier against the traces instead of trusting it.

Every number this project reports comes from a parser written against three harnesses that
record nothing alike, so the parser is the weakest link and has to be tested like one.

Two checks:

1. **Silent truncation.** The traces cut long tool output mid-value with no marker, so a
   run can look like it never measured its final configuration when the measurement was
   simply unreadable. For every run classified "unmeasured", this looks for evaluator
   output after the final edit. If it is there, the verdict is not a finding, it is
   "unknown", and the headline must not count it.

2. **Hand-checkable evidence.** For a random sample it writes the timeline a person needs
   to confirm or refute the verdict by reading the trace: every edit, every launch, every
   measurement, in order, with the final configuration marked.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import (DATA, DIRECT_LAUNCH, INVOKE_SH, NOT_LAUNCH,  # noqa: E402
                     RUN_TARGET, edits_to_target, signature)
from submission import PRIMARY, analyse, readings, scalars, scope_of  # noqa: E402

# Output that came from the evaluator, whether or not a number could be read out of it.
EVALUATOR_OUTPUT = re.compile(r'"profiles"|\bttft\b|\btpot_?p50\b|generation_throughput', re.I)


def timeline(run_dir: Path, meta: dict) -> list[dict]:
    """Every event that bears on the verdict, in trace order."""
    scenario = meta["scenario"]
    script, authored, events = "", set(), []
    for line in (run_dir / "trace.jsonl").open():
        row = json.loads(line)
        i = row.get("i", -1)
        for tool, fragment in edits_to_target(row):
            script = fragment if len(fragment) > 400 and tool in ("write", "heredoc") \
                else f"{script}\n{fragment}"
            events.append({"i": i, "kind": "edit", "config": signature(script), "via": tool})
        inp = row.get("tool_input")
        if isinstance(inp, dict):
            written = str(inp.get("file_path") or inp.get("filePath") or inp.get("path") or "")
            if written.endswith(".sh"):
                authored.add(Path(written).name)
            command = inp.get("command")
            if isinstance(command, str):
                for m in re.finditer(r"cat\s*>\s*(\S*?([\w.-]+\.sh))", command):
                    authored.add(m.group(2))
                if not NOT_LAUNCH.search(command) and (
                        RUN_TARGET.search(command) or DIRECT_LAUNCH.search(command)
                        or any(m.group(2) in authored for m in INVOKE_SH.finditer(command))):
                    events.append({"i": i, "kind": "launch",
                                   "config": signature(script) if script else "unknown"})
        if row.get("type") == "tool_result":
            text = str(row.get("tool_output") or "")
            if EVALUATOR_OUTPUT.search(text):
                # The same scope rule the classifier uses, or the two disagree about what
                # counts as a measurement and the check reports phantom errors.
                scope = scope_of(text)
                values = [] if scope == "smoke" else (readings(text, scenario)
                                                      or scalars(text, scenario))
                kind = ("smoke-test" if scope == "smoke"
                        else "measurement" if values else "unreadable-output")
                events.append({
                    "i": i, "kind": kind, "scope": scope,
                    "config": signature(script) if script else "unknown",
                    "values": [round(v, 6) for v in values],
                    "chars": len(text),
                })
    return events


def truncation_risk(events: list[dict]) -> str:
    """Whether a run's 'unmeasured' verdict could be an artefact of unreadable output."""
    last_edit = max((e["i"] for e in events if e["kind"] == "edit"), default=-1)
    after = [e for e in events if e["i"] > last_edit]
    if any(e["kind"] == "measurement" for e in after):
        return "measured-after-final-edit"     # verdict is wrong: it did measure
    if any(e["kind"] == "unreadable-output" for e in after):
        return "unreadable-after-final-edit"   # verdict is unknown, not a finding
    return "clear"                             # nothing measured after: verdict stands


def main() -> int:
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    manifest = json.loads((DATA / "manifest.json").read_text())
    metas = {r["anon_id"]: r for r in manifest["runs"]}
    rows = json.loads((DATA / "analysis.json").read_text())
    by_run = {r["run"]: r for r in rows}

    print("== Check 1: is 'submitted a configuration it never measured' real?")
    unmeasured = [r for r in rows if r["verdict"] == "unmeasured"]
    verdicts = {}
    for r in unmeasured:
        events = timeline(DATA / "runs" / r["run"], metas[r["run"]])
        verdicts[r["run"]] = truncation_risk(events)
    counts = {k: sum(1 for v in verdicts.values() if v == k) for k in
              ("clear", "unreadable-after-final-edit", "measured-after-final-edit")}
    total = len(unmeasured)
    for k, n in counts.items():
        print(f"   {k:30} {n:3}/{total}")
    survivors = counts["clear"]
    print(f"\n   {survivors} of {total} survive the truncation check.")
    if counts["measured-after-final-edit"]:
        print("   Runs the classifier got wrong: " +
              ", ".join(k for k, v in verdicts.items() if v == "measured-after-final-edit"))

    judged = [r for r in rows if r["verdict"] != "unjudged"]
    print(f"\n   Corrected: of {len(judged)} judgeable runs, "
          f"{survivors} ({survivors / len(judged):.0%}) submitted a configuration they "
          f"provably never measured, where the uncorrected figure was "
          f"{total} ({total / len(judged):.0%}).")

    print(f"\n== Check 2: evidence for {sample_size} random judged runs")
    random.seed(20260904)
    picked = random.sample(judged, min(sample_size, len(judged)))
    out = DATA / "validation"
    out.mkdir(exist_ok=True)
    for r in sorted(picked, key=lambda x: x["run"]):
        events = timeline(DATA / "runs" / r["run"], metas[r["run"]])
        sub = analyse(DATA / "runs" / r["run"], metas[r["run"]])
        lines = [f"# {r['run']}  {r['agent']}  harness={r['harness']}  scenario={r['scenario']}",
                 f"# verdict: {r['verdict']}   truncation: "
                 f"{truncation_risk(events) if r['verdict'] == 'unmeasured' else 'n/a'}",
                 f"# final config: {sub.final_config}",
                 f"# best measured: {sub.best[0] if sub.best else None} = "
                 f"{sub.best[1] if sub.best else None}", ""]
        for e in events:
            extra = f" values={e['values']}" if e.get("values") else \
                    (f" chars={e['chars']}" if e.get("chars") else "")
            lines.append(f"[{e['i']:5}] {e['kind']:20} {e.get('config', '')[:110]}{extra}")
        (out / f"{r['run']}.txt").write_text("\n".join(lines) + "\n")
    print(f"   written to data/validation/  ({len(picked)} files)")
    print("   each holds the ordered timeline a person can check the verdict against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
