# Agent search behaviour

Measuring how coding agents search, from their published trajectories. Two metrics over all 269
InferenceBench runs. No GPU and no API spend: the expensive part was paid for by whoever
generated the traces.

## Why

InferenceBench (arXiv:2607.20468) gives coding agents an H100 and two hours to optimise LLM
serving. Frontier agents reach 8.08x over the PyTorch baseline; a matched-budget random search
reaches 11.53x. The paper locates the deficit in behaviour rather than knowledge, saying the
bottleneck is "the ability to propose diverse configurations, evaluate them systematically, and
submit the best identified solution", and that agents "test only a few distinct configurations
and spend the remaining budget re-measuring, repairing, or optimizing hyperparameters".

That claim is established by reading trajectories by hand. No metric is defined, so nothing
about it can be tracked across models, compared between harnesses, or optimised against.

The 269 published trajectories are Apache-2.0 on Hugging Face and free to analyse. Generating
them cost 100+ H100-hours; reading them costs nothing.

## How it works

`spike/fetch.py` downloads the trajectories. `spike/extract.py` recovers what configurations
each agent tried. `spike/submission.py` asks whether it shipped its own best. `spike/full.py`
runs both over everything and breaks the result down by harness and model.

The obstacle is that configurations never appear as command-line flags. Agents write them into
`start_server.sh` and run that, through a different editing tool per harness: `Edit`/`Write` on
claude, `apply_patch` and narrated diffs on codex, `edit`/`write` on opencode. So the extractor
reconstructs that file across the run and takes a signature at each launch. Signatures normalise
away whitespace, flag order and unrelated edits, so rewriting a comment is not a new idea.

Two findings forced the design:

- **run_0001 never used vLLM.** It hand-rolled a uvicorn server. A parser keyed to vLLM flags
  scores it zero configurations when it tried several.
- **run_0127 launched through `test_server.sh`,** a wrapper it wrote itself. Detecting launches
  by script name would mean guessing names, so the extractor tracks which shell scripts the run
  authored and treats invoking one of those as a launch.

### What made extraction hard

Two runs decided the design. **run_0001 never used vLLM at all**; it hand-rolled a uvicorn
server, so a parser keyed to vLLM flags scores it as trying nothing when it tried several.
**run_0127 launched through `test_server.sh`, a wrapper it wrote itself**, so detecting launches
by script name would mean guessing names. The extractor tracks which shell scripts each run
authored and treats invoking one of those as a launch.

The behaviour the paper describes is visible run by run. run_0016 evaluated 3 distinct
configurations and relaunched them more than twenty times, then failed the quality gate with
empty outputs.

## Results on all 269 runs

Of the 33 runs that measured two or more configurations at comparable scope:

| | |
|---|---|
| submitted the best one they measured | 8 (24%) |
| submitted one within 2% of it | 2 (6%) |
| submitted a measured configuration that was clearly worse | 8 (24%) |
| submitted a configuration they never measured | 15 (45%) |

Restricted to readings whose full scope is provable, 27 runs, the shape holds: 22% submitted
their best, 15% something worse, 56% something unmeasured.

**Roughly seven in ten agents that had a choice to make did not ship the best thing they had
already measured.** The 24% saw a number and shipped something else anyway, by a median of 6%.
The 45% never made the comparison: they edited after their final measurement and stopped. That
second failure is the larger one, and it is not the one the paper names.

A truncation check reduces that 45% to 33% provably never measured, with the remainder unknown
rather than exonerated. See [VALIDATION.md](VALIDATION.md).

run_0138 is the clean case, verified by hand. Its last measurement is at step 214. At step 219
it adds `kv-cache-dtype=fp8` to a configuration it had measured without it. The run ends at step
221. The server that got scored was never benchmarked. run_0006 is the same shape with more
work behind it: fourteen measurements, then a final edit changing two parameters, unmeasured.

**Most runs never had a choice to make.** The median run launched 3 distinct configurations in a
two-hour budget on an H100, and 15 runs yield no recoverable configuration at all. That is the
paper's "test only a few distinct configurations" as a number.

**Searching more did not help.** Runs that submitted their best explored about as many distinct
configurations as runs that did not. Breadth and selection are separate failures, and only the
second is cheap to fix.

### Why the judgeable set is small

236 of 269 runs cannot be judged. Most measured one configuration or none at comparable scope,
which is a fact about the agents; the rest lost their measurements to truncated tool output,
which is a fact about the traces. Both are reported rather than worked around, and the
[validation notes](VALIDATION.md) record what hand-checking changed.

## What is still open

1. **Measurement coverage.** 97 runs have no recoverable measurement, because tool output is
   truncated mid-value. Every run recovered widens the sample the headline rests on.
2. **Baselines.** Outcomes here are raw latency and throughput, not speedup over the PyTorch
   baseline, and the baselines are not in the dataset. Without them a run that never launched
   anything can look good: run_0126 gave up 34 minutes into a two-hour budget, never started a
   server, and reports the best raw number in its scenario because the default answered.
3. **Whether the authors want it.** Not asked yet.

## Running it

```console
python3 spike/fetch.py all    # all 269 runs into data/, ~250 MB
python3 spike/full.py         # both metrics, by harness and by model

python3 spike/fetch.py 4      # or 4 runs per harness, to try it
python3 spike/extract.py      # configurations per run
python3 spike/submission.py   # did the agent ship its own best?
python3 spike/report.py       # configurations joined to raw outcomes
```

No dependencies beyond the standard library, and no GPU. The trajectories are
[Apache-2.0 on Hugging Face](https://huggingface.co/datasets/aisa-group/InferenceBench-Trajectories);
`data/` is gitignored.

## Credit

The trajectories and the benchmark are the work of the InferenceBench authors
([paper](https://arxiv.org/abs/2607.20468), [code](https://github.com/aisa-group/InferenceBench)),
who published 100+ H100-hours of runs under a permissive licence. This repository only reads
what they released.

MIT licensed. Corrections welcome, particularly to the signature function, which is where a
wrong number would come from.

