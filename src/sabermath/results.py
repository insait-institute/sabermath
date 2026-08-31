from __future__ import annotations

import json
import re
from pathlib import Path

from .instructions import INSTRUCTION_KEYS

DEFAULT_RESULTS_DIR = Path("results/evaluation")

# Files that live in the results directory but are not runs.
_NON_RESULT_FILES = frozenset(
    {"query_sample.json", ".provenance.json", "summary.json"}
)

# A pre-p0-naming file: "<model>.json", optionally with a subset/shard/part
# marker but no prompt key. These predate the convention that every run
# records its arm in its filename; the prompt key is recovered from the
# payload where it is recorded, and otherwise assumed to be p0 - which is
# only ever allowed to FILL a cell, never to outrank a canonical run (see
# run_rank).
_LEGACY_NAME_RE = re.compile(
    r"^(?P<model>.+?)"
    r"(?:__n(?P<n>\d+)_seed(?P<seed>\d+))?"
    r"(?:__shard(?P<shard>\d+)of(?P<shards>\d+))?"
    r"(?:__part-(?P<part>.+))?$"
)

_NAME_RE = re.compile(
    r"^(?P<model>.+?)"
    r"__(?P<instruction_key>" + "|".join(INSTRUCTION_KEYS) + r")"
    r"(?:__(?P<protocol_tag>nl1|nl2))?"
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
        "protocol_tag": groups["protocol_tag"] or "",
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
        "protocol_tag": "",
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


def run_rank(payload: dict) -> int:
    _, report = report_of(payload)
    if report is None:
        return 0
    prompt = report.get("prompt")
    if not isinstance(prompt, dict):
        # No prompt block: written before results recorded their protocol,
        # i.e. before the vendor input envelopes were applied.
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
            # Possibly a pre-p0-naming file. Its arm comes from the payload if
            # recorded there, else it is taken as p0 at rank 0.
            parsed = parse_unprompted_name(path.stem)
            if parsed is None or path.name in _NON_RESULT_FILES:
                continue
        if not include_subsets and parsed["subset"]:
            continue
        if not include_parts and (
            parsed["part"] is not None or parsed["shard"] is not None
        ):
            continue
        # The prompt filter is applied AFTER the arm is known: a
        # pre-p0-naming file carries no prompt key in its name, and its arm is
        # recovered from the payload below.

        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        model_key, report = report_of(payload)
        if report is None or ("error" in report and "tasks" not in report):
            continue

        # A pre-p0-naming file carries no prompt key in its name; recover the
        # arm from the payload, falling back to p0. The prompt filter is
        # applied here, once the arm is known either way.
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
