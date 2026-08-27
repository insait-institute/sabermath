"""DIAGNOSTIC: rank1, vLLM (production) vs the HF reference.

    python scripts/diag_rank1_backends.py --stage vllm --out /tmp/r1.json
    python scripts/diag_rank1_backends.py --stage hf   --out /tmp/r1.json

Two stages, two PROCESSES, on purpose: a 32B at vLLM's gpu_memory_utilization
0.9 and the same 32B resident in HF will not coexist, and in-process vLLM
teardown is unreliable. Stage vllm writes its raw outputs to --out; stage hf
reads them back. Run them as consecutive srun steps in one allocation.

rank1-32b reports a math-vs-word statistic of 16.20%, the lowest in the
roster. Rank1Processor is vLLM-NATIVE - it never had an HF implementation to
be migrated from - so unlike nemotron/Octen/RaDeR there was no reference to
check it against. scripts/rank1_hf_processor.py is that reference; this script
is the comparison.

THREE CHECKS, in increasing strictness:

  A. TOKENIZATION. vLLM's own prompt_token_ids vs the HF path's. Must be
     identical ids, not merely similar lengths - everything downstream is
     meaningless if the two models read different prompts.

  B. TEACHER-FORCED. vLLM's OWN generated chain, re-fed to HF, judgment
     distribution compared at the same position. This is the real backend
     test: same tokens in, same tokens out, so any disagreement is numerics.

  C. FREE-RUNNING. HF generates its own chain and is scored end-to-end. This
     is what production would do, but greedy decoding over thousands of tokens
     is chaotic - one near-tie forks the chains permanently. Check C failing
     while B passes means the CHAINS diverged, which is expected, not a
     backend defect. The report prints the first divergence index so the two
     can be told apart rather than guessed at.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from datasets import load_dataset  # noqa: E402

from embed import get_top5_candidates  # noqa: E402

VARIANTS = ("full", "math", "text")


def build_probes(cfg, n):
    targets = load_dataset(cfg["hf_datasets"]["targets_maths_words_fixed"])["train"]
    cands = load_dataset(cfg["hf_datasets"]["candidates_maths_words_fixed"])["train"]
    probes = []
    for i in range(n):
        t = targets[i]
        docs = [cands[s]["problem_fixed"] + cands[s]["solution_fixed"]
                for s in get_top5_candidates(t)]
        probes.append({
            "tid": t["id"],
            "queries": {"full": t["problem_fixed"],
                        "math": t["problem_math_expr"],
                        "text": t["problem_text_only"]},
            "docs": docs,
        })
    return probes


def stage_vllm(args, cfg, probes):
    from sabermath.processors import Rank1Processor

    proc = Rank1Processor(args.model, tensor_parallel_size=1,
                          max_thinking_tokens=args.max_new_tokens)
    proc._init()

    # Flatten to one generate() call so vLLM batches exactly as production
    # does (get_scores passes all of a query's candidates at once).
    flat, prompts = [], []
    for p in probes:
        for v in VARIANTS:
            q = p["queries"][v]
            for ci, doc in enumerate(p["docs"]):
                flat.append((p["tid"], v, ci))
                prompts.append(proc._create_prompt(q, proc._truncate_document(q, doc)))

    print(f"[vLLM] generating {len(prompts)} pairs "
          f"(max_new_tokens={args.max_new_tokens}) ...", flush=True)
    outs = proc._llm.generate(prompts, proc._sampling_params, use_tqdm=True)

    records = []
    for (tid, v, ci), out in zip(flat, outs):
        o = out.outputs[0]
        final = o.logprobs[-1]
        records.append({
            "tid": tid, "variant": v, "cand": ci,
            "prompt_token_ids": list(out.prompt_token_ids),
            "gen_ids": list(o.token_ids),
            "n_gen": len(o.token_ids),
            "n_logprob_steps": len(o.logprobs),
            "finish_reason": o.finish_reason,
            "stop_reason": str(o.stop_reason),
            "score": proc._relevance_score(o),
            "true_token": proc._true_token,
            "false_token": proc._false_token,
            "final_top": {str(k): float(val.logprob) for k, val in final.items()},
            "tail": proc._tokenizer.decode(list(o.token_ids)[-24:],
                                           skip_special_tokens=False),
        })

    Path(args.out).write_text(json.dumps({"model": args.model,
                                          "max_new_tokens": args.max_new_tokens,
                                          "records": records}))
    lens = np.array([r["n_gen"] for r in records])
    trunc = sum(1 for r in records if r["finish_reason"] != "stop")

    # Does logprobs[-1] actually sit on the judgment?
    #
    # _relevance_score reads the LAST position and looks up " true"/" false"
    # there. That is only the judgment position if vLLM kept the stop string's
    # tokens in token_ids/logprobs. If vLLM strips them (the stop string is
    # removed from .text by default), the last position is whatever preceded
    # "</think>" and the production score is being read off the wrong row -
    # which would be a far larger finding than any HF-vs-vLLM delta. Checked
    # explicitly rather than assumed.
    tt, ft = records[0]["true_token"], records[0]["false_token"]
    on_judgment = sum(1 for r in records if r["gen_ids"] and
                      r["gen_ids"][-1] in (tt, ft))
    visible = sum(1 for r in records
                  if str(tt) in r["final_top"] or str(ft) in r["final_top"])
    print(f"\n[vLLM] last generated token is ' true'/' false': "
          f"{on_judgment}/{len(records)}")
    print(f"[vLLM] true or false present in the final top-20: "
          f"{visible}/{len(records)}")
    both = sum(1 for r in records
               if str(tt) in r["final_top"] and str(ft) in r["final_top"])
    print(f"[vLLM] BOTH present (score is a real ratio, not a -1e4 fallback): "
          f"{both}/{len(records)}")
    sc = np.array([r["score"] for r in records])
    print(f"[vLLM] score: min {sc.min():.4f} median {np.median(sc):.4f} "
          f"max {sc.max():.4f} | exactly 0.5 (tie fallback): "
          f"{int(np.sum(sc == 0.5))}/{len(sc)}")
    print(f"\n[vLLM] wrote {len(records)} records -> {args.out}")
    print(f"[vLLM] chain length: min {lens.min()} median {int(np.median(lens))} "
          f"mean {lens.mean():.0f} max {lens.max()}")
    print(f"[vLLM] token_ids vs logprob steps aligned: "
          f"{all(r['n_gen'] == r['n_logprob_steps'] for r in records)}")
    print(f"[vLLM] hit max_new_tokens without a stop string: {trunc}/{len(records)}")
    print(f"[vLLM] sample tail: {records[0]['tail']!r}")


def stage_hf(args, cfg, probes):
    from sabermath.processors import Rank1HFProcessor

    data = json.loads(Path(args.out).read_text())
    records = data["records"]
    qmap = {p["tid"]: p for p in probes}

    proc = Rank1HFProcessor(args.model, max_thinking_tokens=args.max_new_tokens,
                            attn_implementation=args.attn, dtype=args.hf_dtype)
    tag = f"HF/{args.hf_dtype}"
    print(f"HF reference: dtype={args.hf_dtype} attn={args.attn} "
          f"(vLLM production dtype is float16)\n")
    proc._init()

    # --- A. tokenization identity -------------------------------------
    print("=== A. TOKENIZATION ===", flush=True)
    bad = 0
    for r in records:
        p = qmap[r["tid"]]
        ids = proc.prompt_ids(p["queries"][r["variant"]], p["docs"][r["cand"]])
        if ids != r["prompt_token_ids"]:
            bad += 1
            if bad == 1:
                print(f"  MISMATCH {r['tid']}/{r['variant']}/{r['cand']}: "
                      f"hf {len(ids)} ids vs vllm {len(r['prompt_token_ids'])}")
    print(f"  {len(records)-bad}/{len(records)} prompts tokenize identically")
    assert proc._true_token == records[0]["true_token"], "true-token id differs"
    assert proc._false_token == records[0]["false_token"], "false-token id differs"
    print(f"  true/false token ids match: {proc._true_token}/{proc._false_token}")

    # --- B. teacher-forced --------------------------------------------
    print("\n=== B. TEACHER-FORCED (vLLM's own chain, re-scored by HF) ===",
          flush=True)
    fv, fh = [], []
    forced = {}
    for i, r in enumerate(records):
        if len(r["prompt_token_ids"]) + r["n_gen"] > args.max_forced_len:
            continue
        res = proc.score_forced(r["prompt_token_ids"], r["gen_ids"])
        forced[(r["tid"], r["variant"], r["cand"])] = res["score"]
        fv.append(r["score"]); fh.append(res["score"])
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(records)}", flush=True)
    report(f"teacher-forced [vLLM/fp16 vs {tag}]", np.array(fv), np.array(fh))
    verdicts(probes, records, forced, f"teacher-forced [{tag}]")

    if args.n_free <= 0:
        return

    # --- C. free-running ----------------------------------------------
    keep = {p["tid"] for p in probes[:args.n_free]}
    sub = [r for r in records if r["tid"] in keep]
    print(f"\n=== C. FREE-RUNNING (HF generates its own chain, "
          f"{len(sub)} pairs) ===", flush=True)
    gv, gh, first_div = [], [], []
    free = {}
    for i, r in enumerate(sub):
        p = qmap[r["tid"]]
        d = proc._generate_one(p["queries"][r["variant"]], p["docs"][r["cand"]])
        free[(r["tid"], r["variant"], r["cand"])] = d["score"]
        gv.append(r["score"]); gh.append(d["score"])
        a, b = r["gen_ids"], d["gen_ids"]
        k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]),
                 min(len(a), len(b)))
        first_div.append(k if (k < min(len(a), len(b)) or len(a) != len(b)) else -1)
        print(f"  [{i+1}/{len(sub)}] {r['tid'][:14]:<14} {r['variant']:<4} c{r['cand']} "
              f"vllm {r['score']:.4f} hf {d['score']:.4f}  "
              f"len {r['n_gen']}/{d['n_generated']} diverge@{first_div[-1]}",
              flush=True)
    report(f"free-running [vLLM/fp16 vs {tag}]", np.array(gv), np.array(gh))
    fd = np.array([x for x in first_div if x >= 0])
    print(f"  chains identical: {sum(1 for x in first_div if x < 0)}/{len(first_div)}")
    if len(fd):
        print(f"  first divergence index: min {fd.min()} median "
              f"{int(np.median(fd))} max {fd.max()}")
    verdicts(probes[:args.n_free], sub, free, f"free-running [{tag}]")


def report(label, a, b):
    if len(a) < 2:
        print(f"  {label}: too few points ({len(a)})")
        return
    pear = float(np.corrcoef(a, b)[0, 1])
    spear = float(np.corrcoef(np.argsort(np.argsort(a)),
                              np.argsort(np.argsort(b)))[0, 1])
    print(f"\n  --- {label}: agreement over {len(a)} pair scores ---")
    print(f"    Pearson  {pear:.6f}")
    print(f"    Spearman {spear:.6f}")
    print(f"    mean |delta| {np.mean(np.abs(a-b)):.6f}   "
          f"max {np.max(np.abs(a-b)):.6f}")
    print(f"    identical to 1e-4: {int(np.sum(np.abs(a-b) < 1e-4))}/{len(a)}")


def verdicts(probes, records, hf_scores, label):
    """Recompute the math-vs-word verdict under each backend, exactly as
    sim_embeddings.py would: mean score over the 5 candidates, math vs text."""
    v_by = {}
    for r in records:
        v_by[(r["tid"], r["variant"], r["cand"])] = r["score"]

    print(f"\n  --- {label}: math-vs-word verdict per backend ---")
    rows = []
    for p in probes:
        key = lambda v, c: (p["tid"], v, c)  # noqa: E731
        try:
            vm = np.mean([v_by[key("math", c)] for c in range(5)])
            vt = np.mean([v_by[key("text", c)] for c in range(5)])
            hm = np.mean([hf_scores[key("math", c)] for c in range(5)])
            ht = np.mean([hf_scores[key("text", c)] for c in range(5)])
        except KeyError:
            continue
        rows.append((vm > vt, hm > ht))
    if not rows:
        print("    no complete targets")
        return
    print(f"    vLLM  math>text {sum(a for a, _ in rows)}/{len(rows)} = "
          f"{100*sum(a for a, _ in rows)/len(rows):.1f}%")
    print(f"    HF    math>text {sum(b for _, b in rows)}/{len(rows)} = "
          f"{100*sum(b for _, b in rows)/len(rows):.1f}%")
    print(f"    backends agree on {sum(a == b for a, b in rows)}/{len(rows)} verdicts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="both", choices=["vllm", "hf", "both"],
                    help="'both' re-execs this file once per stage: the 32B "
                         "cannot be resident under vLLM and HF at the same "
                         "time, and in-process vLLM teardown is unreliable")
    ap.add_argument("--method", default=None,
                    help="math-vs-word method name (an HF id); overrides "
                         "--model, so run_sims.slurm can drive this script")
    ap.add_argument("--out", default=None, help="JSON handoff file")
    ap.add_argument("--model", default="jhu-clsp/rank1-32b")
    ap.add_argument("--config_file", default="config.yaml")
    ap.add_argument("--n", type=int, default=12, help="targets to probe")
    ap.add_argument("--n_free", type=int, default=3,
                    help="targets for check C; HF generates one chain per "
                         "pair at batch size 1, so this is the expensive part")
    ap.add_argument("--max_new_tokens", type=int, default=8192,
                    help="MUST match on both stages; the vLLM default is 8192 "
                         "and lowering it changes production semantics")
    ap.add_argument("--max_forced_len", type=int, default=12000,
                    help="skip teacher-forcing sequences longer than this "
                         "(one un-chunked forward pass, quadratic attention)")
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--hf_dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"],
                    help="dtype for the HF reference. DEFAULT float16, which "
                         "is what Rank1Processor pins vLLM to - keep it there "
                         "to isolate the BACKEND. rank1-32b's checkpoint is "
                         "bfloat16, so vLLM logs 'Casting torch.bfloat16 to "
                         "torch.float16' on every production run; passing "
                         "bfloat16 here instead answers the DIFFERENT question "
                         "of whether that forced downcast is costing accuracy, "
                         "since bf16's exponent range is far wider than fp16's")
    args = ap.parse_args()

    if args.method:
        args.model = args.method
    if args.out is None:
        args.out = (f"/scratch/{os.environ.get('USER', 'u')}/"
                    f"diag_rank1_{args.model.replace('/', '_')}.json")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.stage == "both":
        # Separate PROCESSES, sequentially: see --stage's help. Stage hf is
        # still attempted if stage vllm fails, so a partial handoff file from
        # an earlier run can still be analysed.
        base = [sys.executable, __file__]
        common = ["--out", args.out, "--model", args.model,
                  "--config_file", args.config_file,
                  "--n", str(args.n), "--n_free", str(args.n_free),
                  "--max_new_tokens", str(args.max_new_tokens),
                  "--max_forced_len", str(args.max_forced_len),
                  "--attn", args.attn, "--hf_dtype", args.hf_dtype]
        for st in ("vllm", "hf"):
            print(f"\n########## STAGE {st} ##########", flush=True)
            rc = subprocess.call(base + ["--stage", st] + common)
            print(f"########## STAGE {st} exited {rc} ##########", flush=True)
            if rc != 0 and st == "vllm":
                print("stage vllm failed; stage hf will reuse any existing "
                      f"handoff at {args.out}", flush=True)
        return

    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)
    probes = build_probes(cfg, args.n)
    print(f"{len(probes)} targets x 3 variants x 5 candidates = "
          f"{len(probes)*15} pairs\n")

    (stage_vllm if args.stage == "vllm" else stage_hf)(args, cfg, probes)


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
