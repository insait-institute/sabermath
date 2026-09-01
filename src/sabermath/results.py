from __future__ import annotations

import json
import re
from pathlib import Path

from .instructions import INSTRUCTION_KEYS

DEFAULT_RESULTS_DIR = Path("results/evaluation")

_NON_RESULT_FILES = frozenset(
    {"query_sample.json", ".provenance.json", "summary.json"}
)

_LEGACY_NAME_RE = re.compile(
    r"^(?P<model>.+?)"
    r"(?:__n(?P<n>\d+)_seed(?P<seed>\d+))?"
    r"(?:__shard(?P<shard>\d+)of(?P<shards>\d+))?"
    r"(?:__part-(?P<part>.+))?$"
)

_NAME_RE = re.compile(
    r"^(?P<model>.+?)"
    r"__(?P<instruction_key>" + "|".join(INSTRUCTION_KEYS) + r")"
    r"(?:__n(?P<n>\d+)_seed(?P<seed>\d+))?"
    r"(?:__shard(?P<shard>\d+)of(?P<shards>\d+))?"
    r"(?:__part-(?P<part>.+))?$"
)


def parse_result_name(stem: str) -> dict | None:
    match = _NAME_RE.match(stem)
    if match is None:
        return None
    groups = match.groupdict()
    n, seed = groups["n"], groups["seed"]
    return {
        "model": groups["model"],
        "instruction_key": groups["instruction_key"],
        "subset": f"n{n}_seed{seed}" if n is not None else "",
        "n": int(n) if n is not None else None,
        "seed": int(seed) if seed is not None else None,
        "shard": int(groups["shard"]) if groups["shard"] is not None else None,
        "shards": int(groups["shards"]) if groups["shards"] is not None else None,
        "part": groups["part"],
    }


def parse_unprompted_name(stem: str) -> dict | None:
    match = _LEGACY_NAME_RE.match(stem)
    if match is None:
        return None
    groups = match.groupdict()
    n, seed = groups["n"], groups["seed"]
    return {
        "model": groups["model"],
        "instruction_key": None,
        "subset": f"n{n}_seed{seed}" if n is not None else "",
        "n": int(n) if n is not None else None,
        "seed": int(seed) if seed is not None else None,
        "shard": int(groups["shard"]) if groups["shard"] is not None else None,
        "shards": int(groups["shards"]) if groups["shards"] is not None else None,
        "part": groups["part"],
    }


def result_name(
    model: str,
    instruction_key: str = "p0",
    subset: str = "full",
    part: str | None = None,
) -> str:
    suffix = "" if subset == "full" else f"__{subset}"
    part_suffix = f"__part-{part}" if part else ""
    return f"{model}__{instruction_key}{suffix}{part_suffix}.json"


def report_of(payload: dict) -> tuple[str | None, dict | None]:
    reports = payload.get("reports") or {}
    if not reports:
        return None, None
    model_key = next(iter(reports))
    return model_key, reports[model_key]


# Canonical-protocol runs that carry no stamp to prove it.
#
# The runner writes the protocol into the report's `prompt` block, and the
# tables show a row only if that block says "canonical" - the gate that keeps
# pre-2026-08-25 numbers out. An early p0 run wrote no `prompt` block at all,
# because p0 has no instruction to record, so three production runs are
# indistinguishable from a pre-protocol one by their own metadata and were
# silently dropped from every generated table.
#
# They are canonical, and each entry says how that is known:
#   - reason-embed-qwen3-8b is the control. Its unstamped file and its later
#     stamped __p0 twin have IDENTICAL per-query nDCG on statement-full, so an
#     unstamped file of this generation is the same computation, not an older
#     one. (It needs no entry here: the stamped twin wins on its own.)
#   - the two below come from the same sweep directory as that control and as
#     retro-star-32b, whose files from that directory ARE stamped canonical,
#     and their own p1-p3 arms are stamped canonical.
#   - rank1-7b's p0 is documented as the current protocol in the header of
#     results/tables/RESULTS_rank1_reasonir_confidence_intervals.md.
#
# Each one reproduces its published statement-full score exactly (0.738,
# 0.664, 0.552). Re-running any of them under the current runner replaces the
# evidence with a stamp; the entry can then go. Three runs, four filenames:
# rank1-7b has both a bare and a __p0 copy, and they must rank alike.
PRE_STAMP_CANONICAL_RUNS = frozenset(
    {
        "reason-rewriter-reason-embed-8b.json",
        "reason-embed-llama-3.1-8b.json",
        "rank1-7b.json",
        "rank1-7b__p0.json",
    }
)


def run_rank(payload: dict) -> int:
    _, report = report_of(payload)
    if report is None:
        return 0
    prompt = report.get("prompt")
    if not isinstance(prompt, dict):
        path = payload.get("_path")
        if path and Path(path).name in PRE_STAMP_CANONICAL_RUNS:
            return 2
        return 0
    return 2 if prompt.get("protocol") == "canonical" else 1


def is_complete(report: dict, task: str) -> tuple[bool, int, int]:
    ndcgs = (report.get("ndcgs_by_task") or {}).get(task) or []
    total = len(ndcgs)
    scored = len([v for v in ndcgs if v is not None])
    return (total > 0 and scored == total, scored, total)


def load_runs(
    scan: Path | str = DEFAULT_RESULTS_DIR,
    prompts: list[str] | None = None,
    include_subsets: bool = False,
    include_parts: bool = False,
) -> dict[tuple[str, str], dict]:
    scan = Path(scan)
    best: dict[tuple[str, str], tuple[tuple, dict]] = {}
    if not scan.is_dir():
        return {}

    for path in sorted(scan.glob("*.json")):
        parsed = parse_result_name(path.stem)
        if parsed is None:
            parsed = parse_unprompted_name(path.stem)
            if parsed is None or path.name in _NON_RESULT_FILES:
                continue
        if not include_subsets and parsed["subset"]:
            continue
        if not include_parts and (
            parsed["part"] is not None or parsed["shard"] is not None
        ):
            continue
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        model_key, report = report_of(payload)
        if report is None or ("error" in report and "tasks" not in report):
            continue

        instruction_key = parsed["instruction_key"]
        if instruction_key is None:
            prompt = report.get("prompt")
            instruction_key = (
                prompt.get("key") if isinstance(prompt, dict) else None
            ) or "p0"
        if prompts is not None and instruction_key not in prompts:
            continue

        scored = sum(
            len([v for v in vals if v is not None])
            for vals in (report.get("ndcgs_by_task") or {}).values()
        )
        key = (model_key, instruction_key)
        rank = (run_rank(payload), scored, path.stat().st_mtime)
        if key not in best or rank > best[key][0]:
            payload["_path"] = str(path)
            payload["_parsed"] = parsed
            best[key] = (rank, payload)

    return {k: v[1] for k, v in best.items()}
