#!/usr/bin/env python3
"""Did the agent submit the best configuration it had already measured?

This is the paper's third clause -- "propose diverse configurations, evaluate them
systematically, and submit the best identified solution" -- and it is the one clause that
needs no external baseline. A run is compared against itself: of the configurations this
agent measured with its own evaluation harness, did it ship the winner?

A failure here is unambiguous and expensive. The agent did the work, saw the number, and
then shipped something worse. No compute is needed to find it, only the trajectory.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import (DATA, DIRECT_LAUNCH, INVOKE_SH, MEASURE, NOT_LAUNCH,  # noqa: E402
                     RUN_TARGET, edits_to_target, signature)

# Two configurations whose measurements differ by less than this are not distinguishable
# by the agent either: benchmark noise on a shared GPU is larger than a fraction of a
# percent. Calling such a run "submitted a worse configuration" would be scoring the agent
# for a decision it had no evidence to make.
TIE = 0.02

# Agents smoke-test constantly. Of 1,423 evaluator outputs across the corpus, 440 ran four
# requests and only 345 ran at the official scale of 64 to 256. A four-request reading is
# not measuring the same thing as a full one and cannot be compared against it: including
# them made one run look like it had found a configuration five times better than anything
# else it tried.
#
# Scope cannot be read from the command, because agents wrap the call in scripts of their
# own ("./eval_quick.sh"). It can be read from the output, which reports request_count in
# 57% of cases. Readings are therefore tagged full, smoke or unknown. Known smoke tests are
# dropped outright. Unknown ones are kept, because dropping them leaves ten judgeable runs
# out of 269, and every headline is reported twice: once over full and unknown together,
# and once over full-scale readings alone.
MIN_REQUESTS = 64
REQUEST_COUNT = re.compile(r'"request_count"\s*:\s*(\d+)')

# Scenario -> (metric path in a profile, whether higher is better).
PRIMARY: dict[str, tuple[tuple[str, ...], bool]] = {
    "A": (("ttft", "p50"), False),
    "B": (("tpot", "p50"), False),
    "C": (("request_throughput_req_per_s",), True),
    "D": (("tpot", "p50"), False),
}


def json_objects(text: str, needle: str) -> list[dict[str, Any]]:
    """Every top-level JSON object in a blob of stdout that contains `needle`.

    Agents print evaluation output surrounded by log lines, so the JSON has to be found by
    brace matching rather than by parsing the whole string. Traces truncate long outputs,
    so this often finds nothing and `readings` handles the remainder.
    """
    out: list[dict[str, Any]] = []
    for start in (m.start() for m in re.finditer(r"\{", text)):
        depth, in_str, esc = 0, False, False
        for i in range(start, min(len(text), start + 200_000)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    if needle in chunk:
                        try:
                            obj = json.loads(chunk)
                        except ValueError:
                            break
                        if isinstance(obj, dict):
                            out.append(obj)
                    break
        if out and len(out) > 40:
            break
    return out


# Tool output in these traces is truncated, frequently in the middle of the evaluator's
# JSON, so a parser that requires a closing brace recovers almost nothing. These read the
# numbers out of whatever text arrived.
PROFILE_SPLIT = re.compile(r'"(burst|poisson|constant|default|[a-z_]+)"\s*:\s*\{')
NESTED = {"ttft", "tpot", "itl"}

# Agents rarely print the evaluator's JSON. They pull the field they care about with a
# one-line python script, so most measurements arrive as a labelled scalar:
#     ttft_p50 0.04740303446305916
#     tpot_p50 0.0103
#     throughput 102.40143672321037
# These are the same numbers by another route, and ignoring them was what left two thirds
# of runs unjudgeable.
SCALAR_KEYS = {
    "A": (r"ttft_?p50", r"ttft"),
    "B": (r"tpot_?p50", r"tpot"),
    "C": (r"req_?throughput", r"request_throughput_req_per_s", r"throughput"),
    "D": (r"tpot_?p50", r"tpot"),
}


def scalars(text: str, scenario: str) -> list[float]:
    """Labelled scalar readings of the scenario's primary metric.

    Only lines where the label is followed by one number are taken, so a python snippet
    that mentions the key while building the print statement contributes nothing. The
    output must also evidence a full-scale evaluation: an agent that prints one extracted
    number and nothing else has not shown what it measured, so the caller tags its scope.
    """
    out: list[float] = []
    for key in SCALAR_KEYS[scenario]:
        for m in re.finditer(rf"^\s*'?{key}'?[\s:=]+([0-9][0-9.eE+-]*)\s*$", text,
                             re.M | re.I):
            try:
                value = float(m.group(1))
            except ValueError:
                continue
            if value > 0:
                out.append(value)
        if out:
            break
    return out


def scope_of(text: str) -> str:
    """full, smoke or unknown, from any request_count the output happens to show."""
    counts = [int(m.group(1)) for m in REQUEST_COUNT.finditer(text)]
    if not counts:
        return "unknown"
    return "full" if max(counts) >= MIN_REQUESTS else "smoke"


def readings(text: str, scenario: str) -> list[float]:
    """The scenario's primary metric from each profile block present in `text`.

    A profile that served nothing is skipped: an agent whose server returned empty bodies
    has not measured a fast configuration, it has measured a broken one.
    """
    path, _ = PRIMARY[scenario]
    key = path[0]
    out: list[float] = []
    marks = [m.start() for m in PROFILE_SPLIT.finditer(text)]
    for i, start in enumerate(marks):
        block = text[start:marks[i + 1] if i + 1 < len(marks) else len(text)]
        count = REQUEST_COUNT.search(block)
        if count and int(count.group(1)) < MIN_REQUESTS:
            continue
        if re.search(r'"success_count"\s*:\s*0\b', block):
            continue
        if re.search(r'"empty_output_count"\s*:\s*[1-9]', block):
            continue
        if key in NESTED:
            m = re.search(rf'"{key}"\s*:\s*\{{[^}}]*?"{path[1]}"\s*:\s*([0-9.eE+-]+)', block, re.S)
        else:
            m = re.search(rf'"{key}"\s*:\s*([0-9.eE+-]+)', block)
        if not m:
            continue
        try:
            value = float(m.group(1))
        except ValueError:
            continue
        if value > 0:
            out.append(value)
    return out


def primary_of(metrics: dict[str, Any], scenario: str) -> float | None:
    """The scenario's primary number, averaged over load profiles, or None."""
    path, _ = PRIMARY[scenario]
    values: list[float] = []
    for profile in (metrics.get("profiles") or {}).values():
        if not isinstance(profile, dict):
            continue
        # A profile that served nothing is not a measurement of anything.
        if profile.get("success_count") == 0 or profile.get("empty_output_count"):
            continue
        node: Any = profile
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, (int, float)) and node > 0:
            values.append(float(node))
    return sum(values) / len(values) if values else None


@dataclass
class Submission:
    run: str
    harness: str
    agent: str
    scenario: str
    measured: list[tuple[str, float]]      # (config signature, primary metric), non-smoke
    full_only: list[tuple[str, float]]     # the subset whose scope is provably full
    final_config: str
    launches: int

    def best_of(self, readings_: list[tuple[str, float]]) -> tuple[str, float] | None:
        if not readings_:
            return None
        _, higher = PRIMARY[self.scenario]
        return (max if higher else min)(readings_, key=lambda p: p[1])

    @property
    def best(self) -> tuple[str, float] | None:
        return self.best_of(self.measured)

    def verdict_over(self, readings_: list[tuple[str, float]]) -> str:
        """The verdict computed over one set of readings, so strict and lenient can differ."""
        distinct = {c for c, _ in readings_}
        best = self.best_of(readings_)
        if not best or len(distinct) < 2:
            return "unjudged"
        if self.final_config == best[0]:
            return "best"
        if self.final_config not in distinct:
            return "unmeasured"
        _, higher = PRIMARY[self.scenario]
        mine = [v for c, v in readings_ if c == self.final_config]
        submitted = (max if higher else min)(mine)
        gap = ((best[1] - submitted) / best[1]) if higher else ((submitted - best[1]) / best[1])
        return "worse" if abs(gap) > TIE else "tied"

    @property
    def verdict_full(self) -> str:
        return self.verdict_over(self.full_only)

    @property
    def verdict(self) -> str:
        """How the submitted configuration relates to what this agent measured.

        "unjudged"   fewer than two measured configurations, so there was no choice
        "best"       submitted the best configuration it measured
        "tied"       submitted one measured within noise of the best, so it lost nothing
        "worse"      submitted a configuration it measured and knew was clearly not the best
        "unmeasured" submitted a configuration it never measured at all

        The last is a separate failure from the paper's clause and arguably a worse one:
        the agent did not pick badly among its results, it shipped something outside them.
        It is also the easy one to mistake for the others, since an agent that keeps
        editing after its final measurement lands here by default.
        """
        return self.verdict_over(self.measured)

    @property
    def regret_raw(self) -> float | None:
        """Fractional gap between the submitted configuration and the best measured one.

        Computed before the tie test, so `verdict` can use it without recursing.
        """
        if not self.best:
            return None
        _, higher = PRIMARY[self.scenario]
        mine = [v for c, v in self.measured if c == self.final_config]
        if not mine:
            return None
        submitted = (max if higher else min)(mine)
        best = self.best[1]
        if not best or not submitted:
            return None
        return (best - submitted) / best if higher else (submitted - best) / best

    @property
    def regret(self) -> float | None:
        """The gap, reported only when the run actually chose worse."""
        return self.regret_raw if self.verdict == "worse" else None


def analyse(run_dir: Path, meta: dict[str, Any]) -> Submission:
    scenario = meta["scenario"]
    script, authored = "", set()
    measured: list[tuple[str, float]] = []
    full_only: list[tuple[str, float]] = []
    launches = 0
    seen: set[tuple[str, float]] = set()       # agents re-read the same results file


    for line in (run_dir / "trace.jsonl").open():
        row = json.loads(line)
        for tool, fragment in edits_to_target(row):
            script = fragment if len(fragment) > 400 and tool in ("write", "heredoc") \
                else f"{script}\n{fragment}"
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
                    launches += 1
        # Measurements cannot be tied to the command that ordered them: agents run the
        # evaluator with --json-output-file and read the file back several turns later,
        # sometimes repeatedly. So any result carrying evaluator output counts, attributed
        # to the configuration live at that point, and identical readings are collapsed.
        if row.get("type") == "tool_result":
            text = str(row.get("tool_output") or "")
            scope = scope_of(text)
            if scope == "smoke":
                continue
            config = signature(script) if script else "unknown"
            if '"profiles"' in text:
                values = [v for obj in json_objects(text, "profiles")
                          if (v := primary_of(obj, scenario)) is not None]
                if not values:
                    profile_values = readings(text, scenario)
                    values = [sum(profile_values) / len(profile_values)] if profile_values else []
            else:
                values = scalars(text, scenario)
            for value in values:
                key = (config, round(value, 9))
                if key not in seen:
                    seen.add(key)
                    measured.append((config, value))
                    if scope == "full":
                        full_only.append((config, value))

    return Submission(run_dir.name, meta["harness"], meta["agent"], scenario,
                      measured, full_only, signature(script) if script else "unknown",
                      launches)


def main() -> int:
    metas = {r["anon_id"]: r for r in json.loads((DATA / "sample.json").read_text())}
    rows = [analyse(DATA / "runs" / r, metas[r]) for r in sorted(metas)
            if (DATA / "runs" / r / "trace.jsonl").exists()]

    print(f"{'run':10} {'agent':22} {'sc':3} {'measured':>8} {'configs':>7} "
          f"{'verdict':>11}  regret")
    for s in rows:
        regret = f"{s.regret:+.1%}" if s.regret is not None else ""
        print(f"{s.run:10} {s.agent:22} {s.scenario:3} {len(s.measured):8} "
              f"{len({c for c, _ in s.measured}):7} {s.verdict:>11}  {regret}")

    judged = [s for s in rows if s.verdict != "unjudged"]
    print(f"\n{len(judged)}/{len(rows)} runs could be judged "
          f"(a run needs two measured configurations to have had a choice).")
    for name in ("best", "worse", "unmeasured"):
        group = [s for s in judged if s.verdict == name]
        if group:
            print(f"  {name:10} {len(group)}: " + ", ".join(s.run for s in group))
    for s in judged:
        if s.verdict == "unmeasured":
            print(f"\n  {s.run} submitted a configuration it never measured:")
            print(f"    best measured  {s.best[0]}")
            print(f"    submitted      {s.final_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
