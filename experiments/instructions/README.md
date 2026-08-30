# Instruction prompts

Does a task instruction help retrieval on SABER-Math? This directory holds the
raw runs behind that question and the script that turns them into tables.

Reproduce the tables with:

```
python experiments/instructions/report.py
```

No GPU, no cluster and no downloads: everything the tables show is recomputed
from `runs/` on every invocation, so the committed tables cannot drift from the
committed data.

## What was run

Each model scores the whole 1000-query benchmark once per prompt. `p0` is the
no-instruction baseline; `p1`, `p2` and `p3` prepend one instruction to the
query. The prompt texts live in `src/sabermath/instructions.py` and the
per-family mechanism (a text prefix, a vendor `<Instruct>` slot, an API
`task_type` enum) in `scripts/instructions/PROTOCOL.md`. Everything here is the
canonical protocol: the model's own vendor input envelope applies, and the
instruction template is `Instruct: {instruction}\nQuery: {query}`.

`RESULTS.md` reports **statement-full**, the paper's headline task. The run
files carry all three tasks, so statement-statement and full-full tables can be
added to `report.py` without re-running anything.

## What is in runs/

289 result files, one per (model, prompt) run, plus `.provenance.json` recording
the cluster path each came from. The bulk is the instruction sweep itself
(`results/instructions_v2`). The rest are the runs that back cells that sweep
does not contain: the ReasonReranker arms, which were executed as five disjoint
query shards because a 32B generative reranker does not finish 1000 queries
inside a wall-clock limit, and the baselines for models added after the sweep
(the Reason-Rewriter composition, ReasonEmbed-Llama-3.1-8B, Diver-GroupRank).

`results/` is gitignored repo-wide, which is why this directory is `runs/`.

## Three things the reporter gets right, each of which was a bug first

**Sharded runs are pooled, not skipped.** A scan that ignores filenames
containing `shard` reports both ReasonReranker rows as having no instruction
arms. They have complete ones - and the stronger of the two, at 0.7556 on p2,
is the highest number anywhere in the benchmark.

**A cell is shown only if all 1000 queries scored.** An interrupted run leaves a
file padded with nulls; averaging the non-null entries yields a number not
comparable to the full runs on its own row. Partial cells are dropped and listed
under "Incomplete", never silently averaged.

**The p0 file is chosen explicitly.** For the Qwen3-Reranker family the ablation's
`__p0` run empties the model's `<Instruct>` slot, while the main experiment's
`<model>.json` fills it with the vendor default - and the two differ by up to
0.011 nDCG. Both parse as p0 and the bare name sorts first, so taking whichever
appears first measures p1-p3 against a different baseline condition than the arms.
The explicit `__pN` file always wins.

Because of that last point the two Qwen3-Reranker rows show a lower `p0` here
(0.6786 and 0.6408) than the main results table reports (0.683 and 0.652). Both
are correct: the table reports the vendor-default configuration, and this
directory reports the prompt-free one, which is the only baseline the p1-p3
deltas can honestly be measured against.

## Rows without arms

Approach Zero, BM25, Jaccard and TF-IDF have no instruction mechanism at all - a
prompt could only reach them as extra query terms - and `INSTRUCTION_EXCLUDED`
in `scripts/run_rerankers.py` refuses to run them instructed. BM25 is there by a
decision rather than a pathology: it alone had no method-specific objection and
was run, scoring 0.4165 -> 0.3568/0.3630/0.3881, but reporting one lexical
baseline with arms and two without is a harness accident rather than a
distinction, so it was excluded with the others and those runs deleted.
