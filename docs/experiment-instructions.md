# Instruction prompts

Does a task instruction help retrieval on SABER-Math? The runs behind that
question are in `results/evaluation/`, alongside every other nDCG run.

Produce the runs, then the tables:

```bash
python scripts/run_experiments.py --models <key> --prompts p0 p1 p2 p3
python -m sabermath.reporting.instruction_statement_full
```

Reporting needs no GPU, no downloads and no cluster: everything the tables show
is recomputed from `results/evaluation/` on every invocation, so a committed
table cannot drift from the committed data.

## The prompts

Defined verbatim in `src/sabermath/instructions.py`:

| Key | Content |
|---|---|
| `p0` | no instruction — the baseline |
| `p1` | "expert mathematical retrieval engine … conceptually close … ideas and solution techniques actually used" |
| `p2` | "identical problem-solving structure and logical steps, even if … different variables, symbols, or … phrasing" |
| `p3` | "Find the most relevant document based on the query." |
| `pm` | the repo's production math instruction — a fifth key, not an arm |

`pm` exists so the Qwen3-Reranker configuration behind the published numbers
stays represented with correct provenance; it is not part of the ablation.

## How an instruction is applied

The instruction wraps the *finished* query, after the task transform:

    Instruct: {instruction}\nQuery: Problem: {statement}\n\nSolution: {solution}

A single newline (every vendor card that documents this mechanism uses one),
query side only. Everything here is the canonical protocol: the model's own
vendor input envelope applies on top.

**Per-family injection**, because "prepend the instruction" is not what every
family actually does — see `docs/protocol.md` for the full table:

- **Bi-encoders** (vLLM/ST/API embedders, RaDeR bi-encoders, ReasonIR,
  inf-retriever-v1-pro): the query-side map above. Document embeddings are
  cached in-process, so one invocation runs all four arms for ~1.05x the cost
  of one.
- **qwen3-reranker-8b/4b/0.6b**: the model's native pair-level `<Instruct>`
  slot, not an additional query wrap — wrapping too would double-instruct.
  `p0` fills the slot with the VENDOR default, so it is a real "no task
  instruction" baseline like every other row.
- **rank1**, **rader-reranker-7b**, **splade**: query-side map. Caveats:
  rader-reranker's SEQ_CLS head never saw instruction text in training (OOD
  input), and SPLADE expands instruction tokens into real sparse query terms
  (lexically noisy by construction).
- **diver-grouprank-32b**: query-side map with `scaffold_reserve_tokens=1280`
  (vs the production 1024) in EVERY arm, so the prefix can never push a
  20-document group past `max_model_len` and the per-document token budget is
  identical in baseline and instructed arms.
- **gte/reason-moderncolbert**: query-side map with `query_length=256` in
  EVERY arm. ColBERT pads queries to `query_length` with mask tokens, so an
  arm-only value would change the query representation as well as the text —
  a deliberate deviation from the checkpoint defaults (48 / 128).
- **inf-x-retriever**: the query-side map feeds the INSTRUCTED query into its
  aligner rewrite; that interaction is part of the experiment. Rewrites are
  logged per instructed query under `results/evaluation/.rewrites/`.

The 32B generative rerankers run p1–p3 on `--task statement-full` only, the
reported setting.

The reported table is **statement-full**, the paper's headline task. The run
files carry all three tasks, so statement-statement and full-full tables can be
added to `sabermath.reporting` without re-running anything.

## What the runs are

One result file per (model, prompt) cell in `results/evaluation/`. The bulk is
the instruction sweep itself. The rest back cells the sweep does not contain:
the ReasonReranker arms, executed as five disjoint query shards because a 32B
generative reranker does not finish 1000 queries inside one sitting, and the
baselines for models added after the sweep (the Reason-Rewriter composition,
ReasonEmbed-Llama-3.1-8B, Diver-GroupRank).

## Three things the reporter gets right, each of which was a bug first

**Sharded runs are pooled, not skipped.** A scan that ignores filenames
containing `shard` reports both ReasonReranker rows as having no instruction
arms. They have complete ones - and the stronger of the two, at 0.7556 on p2,
is the highest number anywhere in the benchmark. (`run_experiments.py
--merge-shards` stitches such a set back into one file; the reporter pools them
either way.)

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
in `src/sabermath/registry.py` makes `run_experiments.py` refuse to run them
instructed. Each is excluded for its own reason: `tf-idf` fits its vocabulary
on the documents (cosine dilution), `jaccard` distorts the token-set union,
and `approach0`'s `_BROKEN_QUERIES` md5 skip-list matches raw query text, so a
rewritten query reintroduces known segfaults. BM25 is there by a
decision rather than a pathology: it alone had no method-specific objection and
was run, scoring 0.4165 -> 0.3568/0.3630/0.3881, but reporting one lexical
baseline with arms and two without is a harness accident rather than a
distinction, so it was excluded with the others and those runs deleted.
