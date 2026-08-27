"""DIAGNOSTIC: encode the same texts through BOTH backends and diff the
vectors directly, instead of inferring a cause from config files.

    python scripts/compare_embeddings.py --method Octen/Octen-Embedding-8B

Loads the two backends SEQUENTIALLY (ST first, freed, then vLLM) so an 8B
model does not have to fit twice, and encodes a ladder of texts from 3 tokens
to a full problem+solution. Prints cosine(ST, vLLM) per text against token
count.

WHY: for Octen-8B the two backends disagree by median 0.074 on the
math-vs-word scores, and six config-level hypotheses have already been ruled
out - precision (fp32 vLLM matches bf16 vLLM to 6.6e-04), pooling (both
resolve LAST), architecture (stock Qwen3Model, auto_map None), missing
modules (the hand-built stack matches modules.json exactly), prompts
(default_prompt_name null) and truncation (both limits far above these
texts). Config archaeology has run out; this looks at the vectors.

READING IT
----------
  * short texts already diverge  -> the model/pooling path itself differs,
    not tokenization. Suspect what LAST-token pooling actually selects.
  * short agree, long diverge    -> tokenization or position handling;
    divergence should then track token count.
  * all agree                    -> the difference is in get_scores, not
    encode, which would point at the candidate/query call shape instead.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

# The experiment dir, for embed.py / load_models.py...
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ...and the repo's src/, for `import sabermath`. load_models.py normally does
# this insert as a side effect of being imported, but this script does not
# import it - and the sabermath import below is INSIDE main(), so a
# module-level import check cannot catch its absence. Confirmed the hard way:
# job 753360 loaded the whole SentenceTransformers stack, encoded every probe
# text, and only then died with "No module named 'sabermath'".
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from datasets import load_dataset  # noqa: E402


def cos(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="Octen/Octen-Embedding-8B")
    ap.add_argument("--config_file", default="config.yaml")
    ap.add_argument("--dtype", default="bfloat16",
                    help="Same dtype BOTH sides - precision is already ruled "
                         "out, so it is held constant rather than varied.")
    args = ap.parse_args()

    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)
    targets = load_dataset(cfg["hf_datasets"]["targets_maths_words_fixed"])["train"]

    # DEGENERATE probes first: with a single token, every pooling mode
    # coincides (mean of one vector == last of one == cls of one). So a
    # disagreement on a 1-token input CANNOT be a pooling difference - it
    # exonerates pooling and points at tokenization or the forward pass.
    # 2- and 3-token probes bracket it: pooling starts to matter there.
    texts = ["4", "x", "the", " ", "42", "x y", "a b c",
             "2+2=4", "Let x be a positive integer."]
    for i in (0, 5, 100):
        t = targets[i]
        texts.append(t["problem_fixed"])
        texts.append(t["problem_math_expr"])
    texts = [t for t in texts if t is not None and t != ""][:14]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.method, trust_remote_code=True)
    ntok = [len(tok(t)["input_ids"]) for t in texts]
    print(f"{len(texts)} probe texts, token counts: {ntok}\n")

    # CONFIRMATION PROBE. This tokenizer appends <|endoftext|> (151643) to
    # every input, and 1_Pooling says lasttoken - so the ST embedding is the
    # hidden state of that APPENDED token, which is the Qwen3-Embedding
    # convention. If vLLM is instead pooling the final CONTENT token, then
    # handing vLLM the text with an explicit trailing <|endoftext|> should
    # make its last position agree with ST's, and cosine should jump to ~1.
    #
    # Guarded, not assumed: the ids are printed so a DOUBLE-appended EOS
    # (which would invalidate the probe) is visible rather than silent.
    eos_str = tok.decode([tok(texts[0])["input_ids"][-1]])
    texts_eos = [t + eos_str for t in texts]
    ntok_eos = [len(tok(t)["input_ids"]) for t in texts_eos]
    print(f"[~] trailing token appended by the tokenizer: {eos_str!r}")
    print(f"[~] ids for {texts[0]!r}      : {tok(texts[0])['input_ids']}")
    print(f"[~] ids for {texts_eos[0]!r}: {tok(texts_eos[0])['input_ids']}")
    print(f"[~] token counts with explicit EOS: {ntok_eos}")
    if any(b - a != 1 for a, b in zip(ntok, ntok_eos)):
        print("[!] appending EOS did not add exactly one token everywhere - "
              "the probe may be double-appending; read the ids above.")
    print()

    # ---- 1. SentenceTransformers, built exactly as the pre-2026-08-26
    # _build_octen_processor did (hand-built stack; the generic ST path
    # crashes on this repo's 2_Normalize kwarg).
    # Imported BEFORE the ST stack is built: this used to be a late import
    # inside the vLLM section, so a missing module wasted a full model load.
    from sabermath.processors import VLLMProcessor

    print("=== loading SentenceTransformers stack ===")
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import modules
    tr = modules.Transformer(
        args.method,
        model_kwargs={"trust_remote_code": True, "dtype": args.dtype},
        config_kwargs={"trust_remote_code": True},
    )
    print(f"[~] ST max_seq_length = {getattr(tr, 'max_seq_length', '?')}")
    # THREE pooling variants over the SAME loaded transformer, so the model is
    # loaded once. If vLLM matches "mean" rather than "lasttoken", it is
    # silently mean-pooling despite logging pooling_type=LAST - which would
    # name the bug precisely.
    dim = tr.get_embedding_dimension()
    st_emb = {}
    for mode in ("lasttoken", "mean", "cls"):
        stack = SentenceTransformer(modules=[
            tr, modules.Pooling(dim, pooling_mode=mode), modules.Normalize()
        ])
        st_emb[mode] = np.asarray(
            stack.encode(texts, show_progress_bar=False), dtype=np.float64
        )
        del stack
    del tr
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # ---- 2. vLLM
    print("=== loading vLLM ===")
    vp = VLLMProcessor.from_huggingface(
        args.method, dtype=args.dtype, gpu_memory_utilization=0.45
    )
    vl_emb = np.asarray(vp.encode(texts, show_progress_bar=False), dtype=np.float64)
    vl_emb_eos = np.asarray(
        vp.encode(texts_eos, show_progress_bar=False), dtype=np.float64
    )

    print(f"\n{'#':>3} {'tok':>4} {'vs last':>9} {'vs mean':>9} {'vs cls':>9} "
          f"{'+EOS vs last':>13}   text")
    rows = []
    for i, (t, n) in enumerate(zip(texts, ntok)):
        c = {m: cos(st_emb[m][i], vl_emb[i]) for m in st_emb}
        c["eos"] = cos(st_emb["lasttoken"][i], vl_emb_eos[i])
        rows.append((n, c))
        print(f"{i:>3} {n:>4} {c['lasttoken']:>9.5f} {c['mean']:>9.5f} "
              f"{c['cls']:>9.5f} {c['eos']:>13.5f}   {t[:36]!r}")

    print("\n--- do the ST pooling modes agree with EACH OTHER? ---")
    print("    (on a 1-token input they must be identical; divergence there")
    print("     would mean the probe is not actually 1 token)")
    for i, n in enumerate(ntok):
        if n <= 3:
            lm = cos(st_emb["lasttoken"][i], st_emb["mean"][i])
            lc = cos(st_emb["lasttoken"][i], st_emb["cls"][i])
            print(f"    {n} tok {texts[i]!r:>16}: last~mean {lm:.6f}  last~cls {lc:.6f}")

    print("\n--- verdict ---")
    one = [c for n, c in rows if n <= 1]
    for mode in ("lasttoken", "mean", "cls"):
        allc = [c[mode] for _, c in rows]
        print(f"  vLLM      vs ST[{mode:<9}]: mean cos {np.mean(allc):.6f}"
              f"  min {min(allc):.6f}")
    eosc = [c["eos"] for _, c in rows]
    print(f"  vLLM+EOS  vs ST[lasttoken]: mean cos {np.mean(eosc):.6f}"
          f"  min {min(eosc):.6f}")
    base = np.mean([c["lasttoken"] for _, c in rows])
    if np.mean(eosc) > 0.999:
        print("  -> CONFIRMED: vLLM omits the tokenizer's trailing EOS, so its "
              "last-token pool lands on the final CONTENT token. Supplying the "
              "EOS explicitly reproduces the SentenceTransformers reference.")
    elif np.mean(eosc) > base + 0.05:
        print(f"  -> PARTIAL: adding EOS improves agreement "
              f"({base:.4f} -> {np.mean(eosc):.4f}) but does not close it; "
              "EOS is part of the story, not all of it.")
    else:
        print(f"  -> NOT the explanation: adding EOS did not help "
              f"({base:.4f} -> {np.mean(eosc):.4f}).")
    if one:
        best = max(("lasttoken", "mean", "cls"), key=lambda m: np.mean([c[m] for _, c in one]))
        v = np.mean([c[best] for _, c in one])
        print(f"\n  ON 1-TOKEN INPUTS (pooling-invariant): best match is "
              f"{best} at cos {v:.6f}")
        if v < 0.999:
            print("  -> pooling is EXONERATED: a 1-token input pools identically "
                  "under every mode, so the divergence is in tokenization or "
                  "the forward pass itself.")
        else:
            print("  -> 1-token inputs agree, so the forward pass matches; the "
                  "divergence appears only once pooling has something to "
                  "choose between.")
    json.dump({"texts": texts, "ntok": ntok,
               "cos": [{k: v for k, v in c.items()} for _, c in rows]},
              open(f"similarities/.embcmp_{args.method.replace('/','_')}.json", "w"))


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
