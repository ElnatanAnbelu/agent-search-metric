# Validating the classifier

Every number here comes from a parser written against three harnesses that record nothing
alike. The parser is the weakest link, so it gets tested like one. `spike/validate.py` runs
the checks; this records what they found.

## What hand-checking caught

A twenty-run sample was read against the raw traces. Four defects surfaced, and each one had
been silently inflating or deflating the headline.

**Smoke tests counted as measurements.** Agents check their work constantly with a handful of
requests. Of 1,423 evaluator outputs in the corpus, 440 ran four requests and only 345 ran at
the official scale of 64 to 256. A four-request reading is not measuring the same thing as a
full one. Including them let run_0130 appear to have found a configuration five times better
than anything else it tried, when that reading came from `./eval_quick.sh`.

Scope cannot be read from the command, because agents wrap the call in scripts of their own.
It can be read from the output, which reports `request_count` in 57% of cases. Readings are now
tagged full, smoke or unknown; smoke is dropped, and every headline is reported twice, once
over full and unknown together and once over provably full-scale readings alone.

**Shell variables collapsed distinct configurations into one.** Agents parameterise the launch
(`--dtype "${DTYPE}"`) and assign the values elsewhere, usually as `DTYPE="${DTYPE:-bfloat16}"`.
A regex that stops at the quote read every such flag as valueless, so run_0006's nine
parameters all became `true` and configurations that differed collapsed to one signature. The
extractor now resolves variables, including the self-referencing default idiom.

**The truncation artefact.** Traces cut long output mid-value with no marker, so a run can look
like it never measured its final configuration when the measurement was merely unreadable. For
every "unmeasured" verdict the validator now checks whether evaluator output appears after the
final edit; if it does, the run is unknown rather than a finding. This removed roughly a third
of that category.

**The validator disagreed with the classifier.** It counted smoke tests as measurements while
the classifier excluded them, so it reported a classifier error on run_0034 that was really its
own. Both now apply the same scope rule, and the check reports zero classifier errors.

## Where it stands

- 15 runs classified "submitted a configuration it never measured"; 11 survive the truncation
  check, 4 are unknown rather than findings.
- 0 runs where the classifier's verdict is contradicted by the trace.
- 20 sampled runs have their full evidence timeline written to `data/validation/`, so any
  verdict can be checked against the trace without rerunning anything.

## What is still unvalidated

The signature function is the remaining risk. Two configurations that differ only in a
parameter outside the known list are treated as identical, and a parameter set through a
mechanism not yet seen would be missed entirely. Nothing checks that a signature matches the
server that actually ran, because the traces do not record it.
