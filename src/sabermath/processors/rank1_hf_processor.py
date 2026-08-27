"""jhu-clsp/rank1-*: the HF-transformers reference path for Rank1Processor.

Rank1Processor is vLLM-NATIVE - unlike the RaDeR and Qwen3 rerankers, it was
never migrated from an HF implementation, so no "before" existed to check its
numbers against. This is that missing reference, written to make the vLLM path
falsifiable rather than to replace it. Production stays on Rank1Processor.

It is a deliberately literal port. Everything that determines a score is taken
from Rank1Processor rather than restated, so the two cannot drift:

  * the prompt is Rank1Processor._create_prompt itself (imported, not copied);
  * dtype is float16, matching Rank1Processor's vLLM `dtype="float16"` - a
    bf16 reference would disagree in the 3rd decimal for reasons that have
    nothing to do with the backend;
  * decoding is greedy (vLLM temperature=0), max_new_tokens=8192, and stops on
    "</think> true" / "</think> false";
  * the score replicates _relevance_score EXACTLY, including its two quirks:
    the distribution is truncated to the top 20 (vLLM's `logprobs=20`) before
    true/false are looked up, and a missing token falls back to logprob -1e4,
    which underflows exp() to 0.0 and yields the 0.5 tie.

WHAT A DISAGREEMENT HERE DOES AND DOESN'T MEAN. Free-running comparison of two
backends on an 8192-token greedy reasoning chain is intrinsically unstable: a
single near-tie anywhere in the chain sends the two decoders down different
paths, after which the final judgments are answering different reasoning and
are not comparable. That is expected behaviour, not a bug. So this class also
exposes score_forced(), which re-feeds vLLM's OWN generated token ids and reads
the final-position distribution. Teacher-forced agreement isolates the
numerics; free-running agreement additionally requires the chains to survive.
Read the two together - free-running disagreement with teacher-forced agreement
means the chains diverged, not that a backend is wrong.

Batching is per-pair by design and should stay that way for reference use:
HF's stop-string criteria is batch-level, so in a padded batch a sequence that
stopped early keeps getting rows appended and `scores[-1]` is no longer its
judgment position. _generate_one runs one sequence at a time for that reason.
"""

import math
from typing import ClassVar

from .base import ModelProcessor
from .rank1_processor import DEFAULT_MAX_MODEL_LEN, DEFAULT_MODEL, Rank1Processor

# vLLM's SamplingParams(logprobs=20): the returned per-position dict holds the
# top 20 tokens, plus the sampled token when it falls outside them.
VLLM_LOGPROBS_TOP_K = 20

STOP_STRINGS = ["</think> true", "</think> false"]

# _relevance_score's stand-in logprob for a token absent from the top-20.
MISSING_LOGPROB = -1e4



def _accepts_dtype() -> bool:
    """True when transformers takes `dtype=` rather than `torch_dtype=`."""
    import transformers

    return int(transformers.__version__.split(".")[0]) >= 5


class Rank1HFProcessor(ModelProcessor):
    processor: ClassVar[str | None] = "rank1-hf"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        dtype: str = "float16",
        max_thinking_tokens: int = 8192,
        attn_implementation: str = "sdpa",
        device_map: str = "auto",
        top_k: int = VLLM_LOGPROBS_TOP_K,
    ) -> None:
        self._model_name = model_name
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._max_thinking_tokens = max_thinking_tokens
        self._attn_implementation = attn_implementation
        self._device_map = device_map
        self._top_k = top_k

        self._model = None
        self._tokenizer = None
        self._true_token = None
        self._false_token = None

    @property
    def model(self) -> str | None:
        return self._model_name

    def _init(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        # Same two lookups as Rank1Processor._init.
        self._true_token = self._tokenizer(
            " true", add_special_tokens=False
        ).input_ids[0]
        self._false_token = self._tokenizer(
            " false", add_special_tokens=False
        ).input_ids[0]

        # transformers 5.x renamed the `torch_dtype` argument to `dtype`
        # (the same rename that hit config keys during the 5.x refactor).
        # Pinning to either name breaks the other env, so pick by signature.
        import inspect

        dtype_kw = (
            "dtype"
            if "dtype"
            in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
            or _accepts_dtype()
            else "torch_dtype"
        )
        model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            attn_implementation=self._attn_implementation,
            device_map=self._device_map,
            **{dtype_kw: getattr(torch, self._dtype)},
        )
        model.eval()
        self._model = model

    # ---- prompt construction: delegated, never restated -------------------

    def _create_prompt(self, query: str, document: str) -> str:
        return Rank1Processor._create_prompt(query, document)

    def _truncate_document(self, query: str, document: str) -> str:
        """Mirror of Rank1Processor._truncate_document. Reproduced rather than
        called because that method reads self._sampling_params, which only
        exists once vLLM has been constructed."""
        reserve = self._max_thinking_tokens + 64
        overhead = len(
            self._tokenizer(
                self._create_prompt(query, ""), add_special_tokens=False
            ).input_ids
        )
        budget = self._max_model_len - reserve - overhead
        doc_ids = self._tokenizer(document, add_special_tokens=False).input_ids
        if budget > 0 and len(doc_ids) > budget:
            document = self._tokenizer.decode(doc_ids[:budget])
        return document

    def prompt_ids(self, query: str, document: str) -> list[int]:
        """The exact ids vLLM would receive. vLLM tokenizes a raw prompt string
        with add_special_tokens=True; Qwen2's tokenizer adds nothing there, but
        the default is kept rather than assumed so the two stay identical."""
        self._init()
        prompt = self._create_prompt(query, self._truncate_document(query, document))
        return self._tokenizer(prompt).input_ids

    # ---- scoring ----------------------------------------------------------

    def _score_from_logits(self, logits, sampled_id=None) -> tuple[float, dict]:
        """Replicate _relevance_score against a full-vocabulary logit row.

        vLLM never sees the full row - it sees the top 20 - so the row is
        truncated the same way before the lookup, or a token sitting at rank
        50 would be scored here and dropped there."""
        import torch

        logprobs = torch.log_softmax(logits.float(), dim=-1)
        top = torch.topk(logprobs, self._top_k)
        visible = {int(i): float(v) for i, v in zip(top.indices, top.values)}
        if sampled_id is not None and int(sampled_id) not in visible:
            visible[int(sampled_id)] = float(logprobs[int(sampled_id)])

        true_logit = visible.get(self._true_token, MISSING_LOGPROB)
        false_logit = visible.get(self._false_token, MISSING_LOGPROB)

        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        denom = true_score + false_score
        score = 0.5 if denom == 0 else true_score / denom
        return score, {
            "true_logprob": true_logit,
            "false_logprob": false_logit,
            "true_visible": self._true_token in visible,
            "false_visible": self._false_token in visible,
        }

    def score_forced(self, prompt_ids: list[int], gen_ids: list[int]) -> dict:
        """Teacher-forced: score vLLM's own chain under this backend.

        vLLM's _relevance_score reads output.logprobs[-1], the distribution
        that PRODUCED the final generated token. Feeding prompt + gen and
        reading position len(prompt)+len(gen)-2 lands on exactly that row."""
        self._init()
        import torch

        if not gen_ids:
            raise ValueError("gen_ids is empty; nothing was generated")

        ids = list(prompt_ids) + list(gen_ids)
        pos = len(prompt_ids) + len(gen_ids) - 2
        t = torch.tensor([ids], device=self._model.device)
        with torch.no_grad():
            out = self._model(t)
        score, detail = self._score_from_logits(out.logits[0, pos], gen_ids[-1])
        return {"score": score, **detail}

    def _generate_one(self, query: str, document: str) -> dict:
        self._init()
        import torch

        ids = self.prompt_ids(query, document)
        t = torch.tensor([ids], device=self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                t,
                attention_mask=torch.ones_like(t),
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=self._max_thinking_tokens,
                stop_strings=STOP_STRINGS,
                tokenizer=self._tokenizer,
                pad_token_id=self._tokenizer.pad_token_id
                or self._tokenizer.eos_token_id,
                repetition_penalty=1.0,
                return_dict_in_generate=True,
                output_logits=True,
            )
        gen_ids = out.sequences[0, len(ids):].tolist()
        # output_LOGITS, not output_scores: `scores` is what the logits
        # processors left behind, so any repetition_penalty / top-k / min-p
        # sitting in the repo's generation_config.json would be baked into the
        # judgment distribution here and absent from vLLM's, turning a config
        # default into a fake backend disagreement. `logits` is the raw head
        # output, which is what vLLM's temperature=0 path reports. The penalty
        # is also pinned to 1.0 so it cannot alter the CHAIN either.
        #
        # logits[-1] is the step that produced the last generated token - the
        # judgment position whenever a stop string was hit. Batch size is 1,
        # so no finished-sequence padding can shift it.
        score, detail = self._score_from_logits(out.logits[-1][0], gen_ids[-1])
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=False)
        return {
            "score": score,
            "gen_ids": gen_ids,
            "n_generated": len(gen_ids),
            "stopped": any(s in text for s in STOP_STRINGS),
            "tail": text[-80:],
            **detail,
        }

    def get_scores(
        self,
        query: str,
        documents: list[str],
        *,
        show_progress_bar: bool = True,
        batch_size: int | None = None,
        **kwargs,
    ) -> list[float]:
        if not documents:
            return []
        self._init()
        return [self._generate_one(query, d)["score"] for d in documents]
