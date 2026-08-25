"""AQ-MedAI/Diver-GroupRank-32B: a groupwise reasoning reranker served via
vLLM.

The model is fed the query together with a small group of candidate documents
(each tagged with a numerical identifier), generates a reasoning chain inside
<reason>...</reason>, and emits per-document integer relevance scores (0-10)
as JSON inside <answer>...</answer>. This follows the officially supplied
inference code from the Hugging Face model card and the reference
implementation in the Diver GitHub repo (Reranker/rerank_groupwise.py at
https://github.com/AQ-MedAI/Diver): same system/user prompts, same vLLM setup
(dtype=bfloat16, max_model_len=32000), same SamplingParams (temperature=0.3,
top_p=0.8, max_tokens=8000), same JSON-repair parsing, same failure handling
(a group whose output can't be parsed contributes score 0 for all its
documents in that epoch), and the same multi-epoch reshuffle-and-average
scheme (their `repeat_num`/`num_epoch`).

Deviations from the reference, all deliberate:
- Shuffling is seeded deterministically per (query, epoch) instead of from
  one shared global `random.seed(666)` stream, so a checkpointed run that
  resumes mid-benchmark regroups each query identically to the original
  attempt (the reference's global stream makes grouping depend on how many
  queries were processed before it).
- SamplingParams gets an explicit `seed` for the same reproducibility reason
  (temperature=0.3 sampling is otherwise nondeterministic across runs). This
  does not change the sampling distribution the reference uses.
- Each document is additionally token-truncated so the full group prompt is
  guaranteed to fit max_model_len minus the generation budget. The reference
  only caps documents at 3000 characters (kept here too, applied first),
  which is enough for BRIGHT's passages but not for SABER-Math's longer
  problem+solution documents in a 20-document group.
- `logprobs=10` is dropped from SamplingParams - the reference requests it
  but never reads it.

Defaults use their eval group size (group_size=20, disjoint groups) with
repeat_num=2 (the reference implementation's default; their BRIGHT eval
script used repeat_num=6 - pass repeat_num=6 to reproduce that exactly, at
3x the generation cost).
"""

import json
import random
import re
import zlib
from typing import ClassVar

from .base import ModelProcessor

DEFAULT_MODEL = "AQ-MedAI/Diver-GroupRank-32B"
DEFAULT_MAX_MODEL_LEN = 32000

SYSTEM_PROMPT = """Your task is to evaluate and rank documents based on how well they help answer the given query. Follow this evaluation priority:
1. PRIMARY: Usefulness & Helpfulness - Does the document provide actionable information, solutions, or direct answers that help address the user's needs?
2. SECONDARY: Relevance - Does the document contain information related to the query topic?

Evaluation Process:
1. First, identify the user's core intent and what kind of help they need from the query
2. For each document, assess:
   - How directly it addresses the user's intent
   - What actionable information or answers it provides
   - How much it helps solve the user's problem or need
3. Compare documents against each other to ensure proper ranking
4. Assign scores that reflect the relative usefulness ranking

Scoring Scale (0-10):
- 9-10: Extremely helpful, directly answers the query with actionable information
- 7-8: Very helpful, provides substantial useful information for the query
- 5-6: Moderately helpful, contains some useful information but incomplete
- 3-4: Minimally helpful, limited useful information despite topic relevance
- 1-2: Barely helpful, mentions related topics but provides little useful information
- 0: Not helpful at all, cannot assist with answering the query
"""

USER_PROMPT = """I will provide you {TOPK} documents, each indicated by a numerical identifier []. Score these documents based on their Usefulness and Relevance to the query.
Query:
{QUERY}

Documents:
{PASSAGES}

## Final Output Format
You must structure your response in exactly two parts: provide your brief reasoning process first, then output final scores in JSON format like below, with document IDs as string keys and integer scores as values for all {TOPK} documents.
The reasoning process and answer are enclosed within <reason> </reason> and <answer> </answer> tags, respectively. Do NOT output anything outside the specified tags. Follow this exact format:
<reason>
[Analyze each document's usefulness and relevance to the query, explaining your scoring rationale]
</reason>
<answer>
```json
{{"[1]": 5, "[2]": 3, "[3]": 8}}
```
</answer>
"""


def extract_group_scores(output_str: str) -> dict:
    """Parse the {"[1]": 5, ...} JSON from a model completion, including the
    reference implementation's repair steps for common formatting mistakes.
    Raises ValueError if no usable JSON can be recovered."""
    if output_str is None:
        raise ValueError("empty output")

    answer_match = re.search(r"<answer>(.*?)</answer>", output_str, re.DOTALL)
    if answer_match:
        output_str = answer_match.group(1).strip()

    try:
        json_matches = re.findall(r"(?:```json\s*)([\s\S]+?)(?:\s*```)", output_str)
        if json_matches:
            json_str = json_matches[-1].strip()
        else:
            json_str = re.findall(r"\{[\s\S]*?\}", output_str)[-1]

        json_str = json_str.strip()
        # Fix keys like "[7]]" -> "[7]"
        json_str = re.sub(r"(\[\d+\])\]", r"\1", json_str)
        # Close a truncated object
        if json_str.count("{") > json_str.count("}"):
            json_str += "}"
        json_str = re.sub(r"\s+", " ", json_str).replace("\n", " ").strip()

        return dict(json.loads(json_str))
    except (IndexError, json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"no valid JSON scores found in output: {e}") from e


class GroupRankProcessor(ModelProcessor):
    processor: ClassVar[str | None] = "grouprank"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        tensor_parallel_size: int = 1,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "bfloat16",
        group_size: int = 20,
        repeat_num: int = 2,
        max_doc_chars: int = 3000,
        max_gen_tokens: int = 8000,
        seed: int = 666,
        scaffold_reserve_tokens: int = 1024,
    ) -> None:
        self._model_name = model_name
        self._tensor_parallel_size = tensor_parallel_size
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        self._dtype = dtype
        self._group_size = group_size
        self._repeat_num = repeat_num
        self._max_doc_chars = max_doc_chars
        self._max_gen_tokens = max_gen_tokens
        self._seed = seed
        self._scaffold_reserve_tokens = scaffold_reserve_tokens

        self._llm = None
        self._sampling_params = None
        self._tokenizer = None
        self._parse_failures = 0
        self._groups_scored = 0

    @property
    def model(self) -> str | None:
        return self._model_name

    def _init(self) -> None:
        if self._llm is not None:
            return

        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError("Please install vllm to use GroupRankProcessor") from e

        self._llm = LLM(
            model=self._model_name,
            dtype=self._dtype,
            gpu_memory_utilization=self._gpu_memory_utilization,
            tensor_parallel_size=self._tensor_parallel_size,
            max_model_len=self._max_model_len,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.8,
            max_tokens=self._max_gen_tokens,
            seed=self._seed,
        )

    def _truncate_doc(self, document: str) -> str:
        """The reference's 3000-char cap plus a token-level guarantee that a
        full group of documents (plus prompt scaffolding and the generation
        budget) fits inside max_model_len."""
        document = re.sub(r"\n+", " ", document)[: self._max_doc_chars]
        # ~1024 tokens covers the system prompt, user template, chat-template
        # wrapping, and per-document "[i]. " markers for a 20-document group.
        budget = (
            self._max_model_len - self._max_gen_tokens - self._scaffold_reserve_tokens
        ) // self._group_size
        doc_ids = self._tokenizer(document, add_special_tokens=False).input_ids
        if len(doc_ids) > budget:
            document = self._tokenizer.decode(doc_ids[:budget])
        return document

    def _group_prompt(self, query: str, doc_texts: list[str]) -> str:
        docs_str = "".join(
            "[{}]. {}\n\n".format(idx + 1, doc_text)
            for idx, doc_text in enumerate(doc_texts)
        )
        group_text = USER_PROMPT.format(
            QUERY=query, PASSAGES=docs_str, TOPK=len(doc_texts)
        )
        return self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": group_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def get_scores(
        self,
        query: str,
        documents: list[str],
        *,
        show_progress_bar: bool = True,
        batch_size: int | None = None,
        **kwargs,
    ) -> list[float]:
        # batch_size slices the GENERATE CALLS over group-prompts (timing
        # harness's 16-in-flight protocol). It deliberately does NOT touch
        # group_size: docs-per-prompt is this model's official SCORING
        # protocol (changing it changes scores), not a batching knob.
        if not documents:
            return []

        self._init()

        doc_texts = [self._truncate_doc(doc) for doc in documents]

        # Build every (epoch, group) prompt for this query and score them in
        # one batched vLLM call. Groups are disjoint slices of a per-epoch
        # shuffle, exactly like the reference with interval == group_size.
        prompts: list[str] = []
        group_doc_idxs: list[list[int]] = []
        query_key = zlib.crc32(query.encode("utf-8"))
        for epoch in range(self._repeat_num):
            rng = random.Random(self._seed * 1_000_003 + query_key * 1_009 + epoch)
            idxs = list(range(len(documents)))
            rng.shuffle(idxs)
            for start in range(0, len(idxs), self._group_size):
                group = idxs[start : start + self._group_size]
                prompts.append(self._group_prompt(query, [doc_texts[i] for i in group]))
                group_doc_idxs.append(group)

        step = batch_size or len(prompts)
        outputs = []
        for start in range(0, len(prompts), step):
            outputs.extend(
                self._llm.generate(
                    prompts[start : start + step],
                    self._sampling_params,
                    use_tqdm=show_progress_bar,
                )
            )

        collected: dict[int, list[float]] = {i: [] for i in range(len(documents))}
        for out, group in zip(outputs, group_doc_idxs):
            self._groups_scored += 1
            text = out.outputs[0].text
            try:
                idx_score = extract_group_scores(text)
            except ValueError as e:
                # Reference behavior: an unparseable group scores 0 for all
                # of its documents in this epoch.
                self._parse_failures += 1
                print(
                    f"[!!] GroupRank parse failure "
                    f"({self._parse_failures}/{self._groups_scored} groups so "
                    f"far): {e}. Tail of output: {text[-200:]!r}"
                )
                for i in group:
                    collected[i].append(0.0)
                continue

            for key, score in idx_score.items():
                try:
                    local = int(str(key).strip().strip("[]")) - 1
                    score = float(score) / 10.0
                except (ValueError, TypeError):
                    continue
                if 0 <= local < len(group):
                    collected[group[local]].append(score)

        # Average a document's scores across the epochs/groups that scored
        # it; a document the model never emitted a score for gets 0 (matches
        # the reference, where such a document simply never enters the
        # ranking).
        return [
            sum(vals) / len(vals) if vals else 0.0
            for vals in (collected[i] for i in range(len(documents)))
        ]
