#!/usr/bin/env python3
"""Recover what configurations an agent actually tried, from its trajectory.

The paper this spike is aimed at says the bottleneck is "the ability to propose diverse
configurations, evaluate them systematically, and submit the best identified solution",
and localises that by reading trajectories by hand. No metric is defined. The obstacle is
that configurations never appear as command-line flags: agents write them into
`start_server.sh` and run that, through a different editing tool in each harness.

So the object to track is that file. Every harness edits it and every harness runs it:

    claude    Edit(file_path, old_string, new_string) / Write(file_path, content)
    codex     apply_patch heredocs, and unified diffs narrated in assistant text
    opencode  write(filePath, content)

This reconstructs the file over the life of a run, takes a configuration signature at each
launch, and counts what was distinct. It reports what it could not reconstruct rather than
guessing, because a run scored "tried one configuration" when the parser simply missed the
edits would be the same failure the metric exists to expose.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TARGET = "start_server.sh"

# Serving engines, in the order a signature should prefer them when a script mentions more
# than one (a script that installs vLLM but launches SGLang is launching SGLang).
ENGINES = (
    ("sglang", re.compile(r"sglang[._]launch_server|sglang\.srt|-m\s+sglang")),
    ("vllm", re.compile(r"vllm\s+serve|vllm\.entrypoints|-m\s+vllm")),
    ("tgi", re.compile(r"text-generation-launcher|text_generation_server")),
    ("trtllm", re.compile(r"trtllm|tensorrt_llm")),
    ("custom", re.compile(r"uvicorn|fastapi|FastAPI|hypercorn")),
)

# Parameters that change serving behaviour. Anything not here is noise for this purpose:
# ports, hosts, log levels and the model id are fixed by the task.
KNOWN = (
    "max-num-seqs", "max_num_seqs",
    "gpu-memory-utilization", "gpu_memory_utilization", "mem-fraction-static",
    "max-model-len", "max_model_len", "context-length",
    "max-num-batched-tokens", "max_num_batched_tokens", "chunked-prefill-size",
    "enable-chunked-prefill", "disable-chunked-prefill",
    "tensor-parallel-size", "tp-size", "pipeline-parallel-size",
    "quantization", "dtype", "kv-cache-dtype", "load-format",
    "enable-prefix-caching", "disable-log-requests", "enforce-eager",
    "speculative-model", "num-speculative-tokens", "cuda-graph-max-bs",
    "scheduling-policy", "swap-space", "block-size", "attention-backend",
    "compilation-config", "torch-compile", "schedule-conservativeness",
)
# Booleans take no value, so a following word is the next statement, not an argument:
# "--enforce-eager\nexport VLLM_..." must not read as enforce-eager=export.
BOOLEAN = {"enable-chunked-prefill", "disable-chunked-prefill", "enable-prefix-caching",
           "disable-log-requests", "enforce-eager", "torch-compile"}
SHELL_WORD = {"export", "exec", "source", "then", "fi", "do", "done", "&&", "||", "\\"}
# Values may be bare, single- or double-quoted. Quoted ones matter: agents parameterise
# the launch with shell variables ("--dtype \"${DTYPE}\"") and a regex that stops at the
# quote reads every such flag as valueless, collapsing distinct configurations into one.
FLAG = re.compile(r"--(" + "|".join(re.escape(k) for k in KNOWN) +
                  r")(?:[=\s]+(\"[^\"]*\"|'[^']*'|[^\s\\\"']+))?")
ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(\"[^\"]*\"|'[^']*'|[^\s#]*)",
                    re.M)
VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)")
ENVVAR = re.compile(r"\b(VLLM_[A-Z0-9_]+|SGLANG_[A-Z0-9_]+|TORCH_COMPILE|DTYPE|ATTENTION_BACKEND)=([^\s\\\"']+)")

RUN_TARGET = re.compile(r"(?:bash|sh|\./|source\s+)\s*\S*start_server\.sh|\bstart_server\.sh\b")
# Agents wrap the launch in scripts of their own (test_server.sh, run.sh, bench.sh), so a
# launch is also any invocation of a shell script this run wrote. Detecting it by name
# would mean guessing the names; tracking what the agent created does not.
INVOKE_SH = re.compile(r"(?:^|[;&|]|\bbash\s+|\bsh\s+|\bsource\s+|\.\/)\s*(\S*?([\w.-]+\.sh))\b")
DIRECT_LAUNCH = re.compile(r"vllm\s+serve|-m\s+vllm\.entrypoints|-m\s+sglang[._]launch_server|text-generation-launcher")
MEASURE = re.compile(r"evaluate\.py|benchmark\.py|benchmark_serving|\bbenchmark\.txt\b")
# A launch that is only a version check or a help page is not a launch.
NOT_LAUNCH = re.compile(r"--help|--version|\|\s*head|grep\b")


@dataclass
class Event:
    i: int
    kind: str                       # edit | launch | measure
    signature: str = ""
    detail: str = ""


@dataclass
class RunAnalysis:
    run: str
    harness: str
    agent: str
    scenario: str
    events: list[Event] = field(default_factory=list)
    unparsed_edits: int = 0
    edit_tools: dict[str, int] = field(default_factory=dict)

    @property
    def launched(self) -> list[str]:
        """Signatures in launch order, dropping consecutive repeats of the same config."""
        out: list[str] = []
        for e in self.events:
            if e.kind == "launch" and e.signature:
                if not out or out[-1] != e.signature:
                    out.append(e.signature)
        return out

    @property
    def distinct(self) -> int:
        return len({s for s in self.launched if s and s != "unknown"})

    @property
    def measures(self) -> int:
        return sum(1 for e in self.events if e.kind == "measure")


def settle(variables: dict[str, str]) -> dict[str, str]:
    """Reduce a variable map to literals.

    Launch scripts are written as NAME="${NAME:-default}", so a variable's recorded value
    is usually an expression naming itself. Taken literally that resolves to nothing and
    every configuration collapses to the same signature; taken as intended, the default is
    the value the run actually used unless the environment overrode it, which nothing in
    these traces does.
    """
    out = dict(variables)
    for _ in range(4):
        changed = False
        for name, value in list(out.items()):
            def swap(m: re.Match[str]) -> str:
                ref = m.group(1) or m.group(3)
                if ref == name:                       # self-reference: take the default
                    return m.group(2) if m.group(2) is not None else ""
                target = out.get(ref, "")
                if VAR.search(target):                # not yet settled
                    return m.group(0)
                return target or (m.group(2) if m.group(2) is not None else m.group(0))
            new = VAR.sub(swap, value)
            if new != value:
                out[name] = new
                changed = True
        if not changed:
            break
    return out


def resolve(value: str, variables: dict[str, str]) -> str:
    """Substitute shell variables in a flag value, honouring ${NAME:-default}."""
    for _ in range(4):                       # defaults can reference other variables
        def swap(m: re.Match[str]) -> str:
            name = m.group(1) or m.group(3)
            return variables.get(name, m.group(2) if m.group(2) is not None else m.group(0))
        new = VAR.sub(swap, value)
        if new == value:
            break
        value = new
    return value


def signature(script: str) -> str:
    """A configuration signature: the engine plus every serving parameter it sets.

    Whitespace, ordering and unrelated edits (comments, echo lines, retry loops) must not
    produce a new signature, or every rewrite of the file would look like a new idea.
    Shell variables are resolved, because a launch written entirely in terms of them is
    otherwise indistinguishable from every other launch written the same way.
    """
    engine = "custom"
    for name, pat in ENGINES:
        if pat.search(script):
            engine = name
            break
    variables = settle({name: v.strip("\"'") for name, v in ASSIGN.findall(script)})
    params: dict[str, str] = {}
    for key, value in FLAG.findall(script):
        name = key.replace("_", "-")
        value = resolve((value or "").strip("\"'"), variables).strip("\"'")
        if name in BOOLEAN or not value or value in SHELL_WORD or value.startswith("-"):
            value = "true"
        params[name] = value
    for key, value in ENVVAR.findall(script):
        params[key] = resolve(value.strip("\"'"), variables).strip("\"'")
    if not params:
        return engine
    body = ",".join(f"{k}={params[k]}" for k in sorted(params))
    return f"{engine}[{body}]"


def edits_to_target(row: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Yield (tool, new_content_or_fragment) for every edit of start_server.sh in a row.

    Each harness edits differently, so this matches on shape rather than on tool name:
    a full-content write, an old/new string replacement, or a patch body.
    """
    inp = row.get("tool_input")
    if isinstance(inp, dict):
        path = str(inp.get("file_path") or inp.get("filePath") or inp.get("path") or "")
        if TARGET in path:
            for key in ("content", "new_string", "newString", "new_str"):
                if isinstance(inp.get(key), str):
                    yield row.get("tool_name") or "write", inp[key]
                    return
        command = inp.get("command")
        if isinstance(command, str) and TARGET in command:
            # Heredocs (cat > start_server.sh <<'EOF' ... EOF) and sed -i edits both carry
            # their new content inline.
            here = re.search(r"<<\s*'?\w+'?\n(.*?)\n\s*\w+\s*$", command, re.S)
            if here:
                yield "heredoc", here.group(1)
            elif re.search(r"\bsed\b.*-i", command):
                yield "sed", command
    text = row.get("text")
    if isinstance(text, str) and TARGET in text and "@@" in text:
        # codex narrates unified diffs in assistant text; added lines carry the new config.
        added = "\n".join(m[1:] for m in re.findall(r"^\+(?!\+\+).*$", text, re.M))
        if added.strip():
            yield "diff", added


def analyse(run_dir: Path, meta: dict[str, Any]) -> RunAnalysis:
    a = RunAnalysis(run_dir.name, meta.get("harness", "?"), meta.get("agent", "?"),
                    meta.get("scenario", "?"))
    script = ""
    authored: set[str] = set()          # shell scripts this run wrote, by basename
    for line in (run_dir / "trace.jsonl").open():
        row = json.loads(line)
        for tool, fragment in edits_to_target(row):
            a.edit_tools[tool] = a.edit_tools.get(tool, 0) + 1
            # A full write replaces the file; a fragment only adds to what is known. Both
            # feed the same signature, and a fragment that sets no parameter is ignored
            # rather than counted as a configuration.
            script = fragment if len(fragment) > 400 and tool in ("write", "heredoc") \
                else f"{script}\n{fragment}"
            a.events.append(Event(row.get("i", -1), "edit", signature(script), tool))
        inp = row.get("tool_input")
        if isinstance(inp, dict):
            written = str(inp.get("file_path") or inp.get("filePath") or inp.get("path") or "")
            if written.endswith(".sh"):
                authored.add(Path(written).name)
            cmd = inp.get("command")
            if isinstance(cmd, str):
                for m in re.finditer(r"cat\s*>\s*(\S*?([\w.-]+\.sh))", cmd):
                    authored.add(m.group(2))
        command = inp.get("command") if isinstance(inp, dict) else None
        if not isinstance(command, str):
            continue
        if MEASURE.search(command):
            a.events.append(Event(row.get("i", -1), "measure"))
        if NOT_LAUNCH.search(command):
            continue
        if RUN_TARGET.search(command):
            sig = signature(f"{script}\n{command}") if script else "unknown"
            a.events.append(Event(row.get("i", -1), "launch", sig, "script"))
        elif DIRECT_LAUNCH.search(command):
            a.events.append(Event(row.get("i", -1), "launch", signature(command), "direct"))
        elif any(m.group(2) in authored for m in INVOKE_SH.finditer(command)):
            sig = signature(f"{script}\n{command}") if script else "unknown"
            a.events.append(Event(row.get("i", -1), "launch", sig, "wrapper"))
    a.unparsed_edits = sum(1 for e in a.events if e.kind == "launch" and e.signature == "unknown")
    return a


def main() -> int:
    sample = json.loads((DATA / "sample.json").read_text())
    metas = {r["anon_id"]: r for r in sample}
    print(f"{'run':10} {'harness':9} {'agent':22} {'sc':3} {'launch':>6} {'distinct':>8} "
          f"{'measure':>7} {'unknown':>7}  edit tools")
    rows = []
    for run_id in sorted(metas):
        d = DATA / "runs" / run_id
        if not (d / "trace.jsonl").exists():
            continue
        a = analyse(d, metas[run_id])
        rows.append(a)
        tools = ",".join(f"{k}:{v}" for k, v in sorted(a.edit_tools.items())) or "none"
        print(f"{a.run:10} {a.harness:9} {a.agent:22} {a.scenario:3} {len(a.launched):6} "
              f"{a.distinct:8} {a.measures:7} {a.unparsed_edits:7}  {tools}")
    covered = [a for a in rows if a.distinct]
    print(f"\n{len(covered)}/{len(rows)} runs yielded at least one parsed configuration")
    if covered:
        print("distinct configurations per run: "
              f"min {min(a.distinct for a in covered)}, "
              f"max {max(a.distinct for a in covered)}, "
              f"median {sorted(a.distinct for a in covered)[len(covered) // 2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
