"""DIAGNOSTIC: is the ST-vs-vLLM divergence in TOKENIZATION, the FORWARD
PASS, or SCORING?

    python scripts/diag_tokenization.py --method Octen/Octen-Embedding-8B

Three questions, answered in order, so each one is settled before the next
is asked:

  A. TOKENIZATION IDENTITY. Print the exact input ids both sides use - HF's
     tokenizer (what SentenceTransformers feeds) and vLLM's own tokenizer.
     If they differ, that is the answer. If they are identical, tokenization
     is conclusively exonerated and B applies.

  B. FORWARD PASS. Re-embed through vLLM with EXPLICIT prompt_token_ids set
     to the HF ids, bypassing vLLM's tokenization entirely. Both sides then
     provably see the same integers. If agreement is still ~0.69, the
     divergence is inside the model forward or the pooler, not the input.

  C. SCORING. The math-vs-word numbers are cosines of embeddings, not raw
     embeddings, so confirm the score gap follows from the embedding gap
     rather than from anything in get_scores(). Scores are recomputed here
     from the vectors the same way EmbeddingProcessor does.

WHY THIS EXISTS: for Octen-8B three hypotheses have already been tested and
rejected - dtype (fp32 vLLM matches bf16 vLLM to 6.5e-04), pooling mode
(vLLM matches ST[lasttoken] at 0.69, ST[mean] at 0.56, ST[cls] at 0.10, so
LAST is right), and a missing trailing EOS (the probe double-appended, which
its own id dump revealed - it was invalid, not disconfirming). This narrows
the remaining space instead of guessing again.
"""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from datasets import load_dataset  # noqa: E402

from embed import get_top5_candidates  # noqa: E402


def cos(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="Octen/Octen-Embedding-8B")
    ap.add_argument("--config_file", default="config.yaml")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--st-model-kwargs", default=None,
                    help="Extra JSON merged into the ST model_kwargs. Needed "
                         "for models whose remote code demands a setting at "
                         "load time - e.g. jina-v5 raises 'Task must be "
                         "specified before encoding data' without "
                         "default_task, so its TABLE_MODELS entry passes "
                         "{\"default_task\": \"retrieval\"}.")
    ap.add_argument("--vllm-gpu-util", type=float, default=0.45,
                    help="vLLM gpu_memory_utilization. 0.45 suffices up to "
                         "~12B, but 27B in bf16 leaves only ~9 GiB for KV "
                         "cache while vLLM sizes that for the model's FULL "
                         "context - harrier-27b failed needing 17.98 GiB. The "
                         "ST stack is freed before vLLM loads, so raising "
                         "this is safe.")
    ap.add_argument("--vllm-max-len", type=int, default=None,
                    help="Cap vLLM max_model_len. Purely a KV-cache lever: "
                         "the probes top out near 198 tokens, so any cap well "
                         "above that cannot truncate anything and cannot "
                         "change an embedding.")
    ap.add_argument("--raw-text-transformer", action="store_true",
                    help="Gemma3 family: bypass AutoProcessor by driving "
                         "AutoModel/AutoTokenizer directly, reproducing the "
                         "old _build_gemma3_text_embedding_processor. Without "
                         "it these models have NO loadable ST side at all.")
    ap.add_argument("--disable-st-default-prompt", action="store_true",
                    help="Clear sentence-transformers' default_prompt_name "
                         "after loading, as _build_spec_processor does for "
                         "specs carrying disable_st_default_prompt. jina-v5 "
                         "ships default_prompt_name='document', so WITHOUT "
                         "this the ST side silently prepends 'Document: ' to "
                         "every text - including queries - and the comparison "
                         "would measure that bug instead of the backend.")
    args = ap.parse_args()

    import json as _json
    from sabermath.processors import VLLMProcessor

    # Extra model_kwargs demanded by a model's own remote code at load time.
    # jina-v5 raises "Task must be specified before encoding data" without
    # default_task, which is why its TABLE_MODELS entry carries it.
    extra_mk = _json.loads(args.st_model_kwargs) if args.st_model_kwargs else {}

    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)
    targets = load_dataset(cfg["hf_datasets"]["targets_maths_words_fixed"])["train"]
    cands_ds = load_dataset(cfg["hf_datasets"]["candidates_maths_words_fixed"])["train"]

    probes = ["4", "x", "2+2=4", "Let x be a positive integer."]
    tgt = targets[0]
    query = tgt["problem_fixed"]
    docs = [cands_ds[s]["problem_fixed"] + cands_ds[s]["solution_fixed"]
            for s in get_top5_candidates(tgt)]
    all_texts = probes + [query] + docs

    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(args.method, trust_remote_code=True)
    hf_ids = [hf_tok(t)["input_ids"] for t in all_texts]

    # ---- A. SentenceTransformers side -------------------------------------
    print("=== loading SentenceTransformers ===")
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import modules
    # Load the way THIS model's pre-delegation path did, or the comparison is
    # against a stack the model never actually ran under:
    #   * most models used the generic SentenceTransformersProcessor
    #     .from_huggingface(name, trust_remote_code=True), i.e. the full stack
    #     from modules.json;
    #   * Octen-4B/8B could NOT - the generic load raises on their
    #     2_Normalize/config.json ("Normalize.__init__() got an unexpected
    #     keyword argument 'normalize_embeddings'"), which is exactly why
    #     _build_octen_processor hand-built the stack.
    # Try the generic path first and fall back, reporting which was used, so a
    # hand-built stack is never silently compared against a model that would
    # have loaded generically.
    st = None
    try:
        st = SentenceTransformer(
            args.method, trust_remote_code=True,
            model_kwargs={"dtype": args.dtype, **extra_mk},
        )
        print("[~] ST built via the GENERIC path (modules.json)")
        print(f"[~] modules: {[type(m).__name__ for m in st]}")
        if args.disable_st_default_prompt:
            cur = getattr(st, "default_prompt_name", None)
            if cur is not None:
                print(f"[~] clearing default_prompt_name={cur!r}")
                st.default_prompt_name = None
    except Exception as e:
        print(f"[~] generic ST load failed ({type(e).__name__}: {e}); "
              "falling back to the hand-built stack")
    if st is None and args.raw_text_transformer:
        # Gemma3-family fallback, ported from the pre-2026-08-26
        # _build_gemma3_text_embedding_processor. Needed because
        # modules.Transformer unconditionally calls
        # AutoProcessor.from_pretrained(), which routes Gemma3 to a MULTIMODAL
        # image-processor loader that these text-only repos do not ship
        # ("OSError: Can't load image processor for ..."). That blocker is
        # still live under sentence-transformers 5.7.0 - confirmed on
        # KaLM-Embedding-Gemma3-12B-2511, where BOTH the generic load and a
        # plain modules.Transformer fallback die with it. This bypasses
        # AutoProcessor entirely by driving AutoModel/AutoTokenizer directly,
        # which is the only way to get an ST side for these models at all.
        print("[~] building the _RawTextTransformer stack (Gemma3 bypass)")
        from transformers import AutoModel, AutoTokenizer

        class _RawTextTransformer(modules.InputModule):
            save_in_root = True

            def __init__(self):
                super().__init__()
                self._auto_model = AutoModel.from_pretrained(
                    args.method, trust_remote_code=True, dtype=args.dtype
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    args.method, trust_remote_code=True
                )

            def preprocess(self, inputs, prompt=None, **kwargs):
                if prompt:
                    inputs = self._prepend_prompt(inputs, prompt)
                return dict(self.tokenizer(
                    list(inputs), padding=True, truncation=True,
                    return_tensors="pt"))

            def forward(self, features, **kwargs):
                mi = {k: v for k, v in features.items()
                      if k in ("input_ids", "attention_mask", "token_type_ids")}
                features["token_embeddings"] = self._auto_model(**mi).last_hidden_state
                return features

            def get_embedding_dimension(self) -> int:
                return self._auto_model.config.hidden_size

            def save(self, output_path, *a, **k):
                raise NotImplementedError("diagnostic only")

        _tr = _RawTextTransformer()
        st = SentenceTransformer(modules=[
            _tr, modules.Pooling(_tr.get_embedding_dimension(),
                                 pooling_mode="lasttoken"),
            modules.Normalize(),
        ])
        print("[~] ST built via _RawTextTransformer + Pooling(lasttoken) + Normalize")

    if st is None:
        import json as _json
        pool_cfg = {}
        try:
            from huggingface_hub import hf_hub_download
            pool_cfg = _json.load(open(hf_hub_download(
                args.method, "1_Pooling/config.json")))
        except Exception:
            pass
        mode = next((k.replace("pooling_mode_", "").replace("_tokens", "")
                     for k, v in pool_cfg.items()
                     if k.startswith("pooling_mode_") and v is True), "lasttoken")
        mode = {"lasttoken": "lasttoken", "mean": "mean", "cls_token": "cls"}.get(mode, mode)
        print(f"[~] hand-built stack, pooling_mode={mode!r} (from 1_Pooling)")
        tr = modules.Transformer(
            args.method,
            model_kwargs={"trust_remote_code": True, "dtype": args.dtype},
            config_kwargs={"trust_remote_code": True},
        )
        st = SentenceTransformer(modules=[
            tr, modules.Pooling(tr.get_embedding_dimension(), pooling_mode=mode),
            modules.Normalize(),
        ])
    st_emb = np.asarray(st.encode(all_texts, show_progress_bar=False), dtype=np.float64)
    # keep the stack alive only long enough for probe D (6 short encodes)
    st_solo = st
    st_single_raw = [st_solo.encode([t], show_progress_bar=False)[0]
                     for t in all_texts[:6]]
    # Drop EVERY reference to the transformer, not just the SentenceTransformer
    # wrapper. The generic branch binds it to `tr` and the Gemma3 branch to
    # `_tr`; either one left alive keeps the whole model resident and vLLM then
    # starts against an already-full GPU. Invisible up to ~12B, fatal at 27B -
    # harrier-oss-v1-27b failed with "Free memory on device cuda:0
    # (88.26/139.8 GiB) ... less than desired GPU memory utilization (0.8)",
    # and the ~51 GiB missing was exactly this leak.
    # Rebinding, not del: `tr`/`_tr` may or may not exist depending on which
    # branch built the stack, and mutating locals() does nothing in CPython.
    # Assigning None is what actually drops the reference in both cases.
    del st
    st_solo = None
    tr = None
    _tr = None
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    # ---- vLLM -------------------------------------------------------------
    print("=== loading vLLM ===")
    vkw = {"dtype": args.dtype, "gpu_memory_utilization": args.vllm_gpu_util}
    if args.vllm_max_len:
        vkw["max_model_len"] = args.vllm_max_len
    print(f"[~] vLLM kwargs: {vkw}")
    vp = VLLMProcessor.from_huggingface(args.method, **vkw)
    vl_text = np.asarray(vp.encode(all_texts, show_progress_bar=False), dtype=np.float64)

    print("\n===== A. TOKENIZATION IDENTITY =====")
    v_tok = vp._llm.get_tokenizer()
    same = True
    for t, ids in zip(all_texts[:4], hf_ids[:4]):
        vids = v_tok(t)["input_ids"]
        ok = vids == ids
        same &= ok
        print(f"  {t!r:>32}  HF={ids}")
        print(f"  {'':>32}  vLLM={vids}   {'MATCH' if ok else '*** DIFFERENT ***'}")
    print(f"  -> tokenization identical on probes: {same}")

    # ---- B. forced token ids ---------------------------------------------
    print("\n===== B. FORWARD PASS (vLLM fed HF's exact ids) =====")
    outs = vp._llm.embed([{"prompt_token_ids": ids} for ids in hf_ids], use_tqdm=False)
    vl_ids = np.asarray([o.outputs.embedding for o in outs], dtype=np.float64)
    print(f"  {'text':>32} {'vLLM(text)':>11} {'vLLM(ids)':>11}")
    for i, t in enumerate(all_texts[:6]):
        print(f"  {t[:32]!r:>32} {cos(st_emb[i], vl_text[i]):>11.5f} "
              f"{cos(st_emb[i], vl_ids[i]):>11.5f}")
    mt = np.mean([cos(st_emb[i], vl_text[i]) for i in range(len(all_texts))])
    mi = np.mean([cos(st_emb[i], vl_ids[i]) for i in range(len(all_texts))])
    print(f"  mean cos vs ST[lasttoken]:  via text {mt:.5f}   via ids {mi:.5f}")
    if mi > 0.999:
        print("  -> BACKENDS AGREE: vLLM reproduces the ST reference. Nothing "
              "to explain - tokenization, forward pass and pooler all match.")
    elif same:
        print("  -> FORWARD PASS or POOLER: ids are identical AND forcing them "
              "changes nothing, so the input is not the problem.")
    else:
        print("  -> tokenization differs; see A.")

    # ---- D. batching / padding -------------------------------------------
    # Cheap and decisive for the padding hypothesis. Everything above encodes
    # all texts in ONE batch, so a 2-token probe is padded out to the longest
    # text in the set. If a lasttoken pooler takes the literal final position
    # rather than the last NON-PAD position, short texts pool a PAD token -
    # which fits Octen's length pattern ('4' 0.56, 'x y' 0.85, long 0.67-0.78)
    # and would mean the hand-built stack, not vLLM, is the broken side.
    # Encoding one text at a time removes padding entirely.
    print("\n===== D. BATCHED vs ONE-AT-A-TIME (padding probe) =====")
    st_single = np.asarray(st_single_raw, dtype=np.float64)
    if True:
        print(f"  {'text':>32} {'batched':>9} {'solo':>9}")
        for i, t in enumerate(all_texts[:6]):
            print(f"  {t[:32]!r:>32} {cos(st_emb[i], vl_text[i]):>9.5f} "
                  f"{cos(st_single[i], vl_text[i]):>9.5f}")
        mb = np.mean([cos(st_emb[i], vl_text[i]) for i in range(6)])
        ms = np.mean([cos(st_single[i], vl_text[i]) for i in range(6)])
        print(f"  mean vs vLLM: batched {mb:.5f}   solo {ms:.5f}")
        if ms > 0.999 and mb < 0.99:
            print("  -> PADDING: the ST stack mis-pools padded batches. vLLM is "
                  "correct and the ST-derived baseline is the broken number.")
        elif abs(ms - mb) < 0.01:
            print("  -> not padding: batching makes no difference.")

    # ---- C. scoring -------------------------------------------------------
    print("\n===== C. SCORING (cosine, as EmbeddingProcessor computes it) =====")
    qi = len(probes)
    di = list(range(qi + 1, qi + 1 + len(docs)))
    for name, E in (("ST  ", st_emb), ("vLLM", vl_text)):
        sc = [cos(E[qi], E[j]) for j in di]
        print(f"  {name} scores: {[round(x,4) for x in sc]}  mean {np.mean(sc):.4f}")
    print("  (embeddings already differ, so a score gap here is a CONSEQUENCE,")
    print("   not an independent scoring bug)")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
