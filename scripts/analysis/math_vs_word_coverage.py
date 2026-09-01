#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from sabermath.math_vs_word import SIMILARITIES_DIR
from sabermath.math_vs_word.aggregate import NON_EMBEDDING_METHODS
from sabermath.math_vs_word.load_models import get_scores_kwargs
from sabermath.math_vs_word.roster import (
    LOAD_MODELS_PY,
    REGISTRY_PY,
    allowed_models,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

EXPECTED_TARGETS = 969
SIM_FIELDS = (
    "pr_full_vs_candidates",
    "pr_math_vs_candidates",
    "pr_text_vs_candidates",
)

def envelope_for(method: str) -> dict | None:
    try:
        return dict(get_scores_kwargs(method, "p0"))
    except Exception:
        return None

def inspect(method: str, sim_dir: Path, arm: str | None = None) -> tuple[str, str]:
    stem = method.replace("/", "_")
    if arm:
        stem = f"{stem}__{arm}"
    path = sim_dir / f"{stem}.json"
    if not path.exists():
        return "missing", "no similarities file"
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return "bad", f"unreadable ({type(exc).__name__})"
    n = len(payload)
    nulls = sum(
        1
        for entry in payload.values()
        if any(entry.get(f) is None for f in SIM_FIELDS)
    )
    if nulls:
        return "null", f"{n} targets, {nulls} with a null similarity"
    if n < EXPECTED_TARGETS:
        return "short", f"{n}/{EXPECTED_TARGETS} targets - preempted mid-run"
    if n > EXPECTED_TARGETS:
        return "bad", f"{n} targets, expected {EXPECTED_TARGETS}"
    return "ok", f"{n}/{EXPECTED_TARGETS}"

def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-dir", type=Path, default=SIMILARITIES_DIR)
    parser.add_argument("--load-models", type=Path, default=LOAD_MODELS_PY)
    parser.add_argument(
        "--extra-source",
        type=Path,
        nargs="*",
        default=[REGISTRY_PY],
        help="Modules whose top-level literals load_models.py imports.",
    )
    parser.add_argument(
        "--emit-commands",
        action="store_true",
        help="Print the calc_sims.py command for each incomplete method.",
    )
    parser.add_argument(
        "--arms",
        nargs="*",
        default=None,
        help="Instruction-ablation arms to check as well (e.g. p0 p1 p2 p3). "
        "Each is stored as similarities/<method>__<arm>.json. Only embedding "
        "methods have these; the lexical methods are instruction controls.",
    )
    args = parser.parse_args(argv)

    methods = allowed_models(args.load_models, args.extra_source) + NON_EMBEDDING_METHODS
    results = [(m, *inspect(m, args.sim_dir)) for m in methods]

    complete = [r for r in results if r[1] == "ok"]
    broken = [r for r in results if r[1] != "ok"]

    print(f"{len(methods)} runnable methods, {len(complete)} complete.\n")
    for method, status, detail in broken:
        print(f"  [{status.upper():>7}] {method} - {detail}")

    known = {m.replace("/", "_") for m in methods}
    orphans = sorted(
        p.stem for p in args.sim_dir.glob("*.json") if p.stem not in known
    )
    if orphans:
        print(f"\n{len(orphans)} similarity file(s) with no method in "
              f"ALLOWED_MODELS (stale, or a roster entry was renamed):")
        for stem in orphans:
            print(f"  {stem}")

    if broken and args.emit_commands:
        print("\n# Re-run the incomplete methods. A 'short' file resumes from\n"
              "# where it stopped; --force_recalc restarts from scratch, which\n"
              "# is what a 'null' or 'bad' file needs.")
        for method, status, _ in broken:
            force = " --force_recalc" if status in ("null", "bad") else ""
            print(
                f"python scripts/analysis/math_vs_word.py "
                f"--method {method!r}{force}"
            )

    if args.arms:
        embedding_methods = [m for m in methods if m not in NON_EMBEDDING_METHODS]
        print(f"\n{'=' * 68}\nInstruction ablation: "
              f"{len(embedding_methods)} embedding methods x {len(args.arms)} arms")
        reusable, to_run, unknown = [], [], []
        for arm in args.arms:
            for method in embedding_methods:
                status, _ = inspect(method, args.sim_dir, arm)
                if status == "ok":
                    continue
                # p0 is the default arm, so it is always a copy, never a run.
                if arm == "p0":
                    if envelope_for(method) is None:
                        unknown.append((method, arm))
                    else:
                        reusable.append(method)
                else:
                    to_run.append((method, arm))

        print(f"  needs a run : {len(to_run)}")
        print(f"  p0 reusable : {len(reusable)}  (p0 IS the default arm - the "
              f"file already on disk is that run)")
        if unknown:
            print(f"  UNRESOLVED  : {len(unknown)}  (no registry model key - "
                  f"envelope unknown, so p0 cannot be assumed reusable)")
            for method, arm in unknown:
                print(f"      {method} / {arm}")

        if args.emit_commands:
            if reusable:
                print("\n# p0 arms that are the default run under another name.")
                print("# Copy rather than recompute.")
                for method in sorted(set(reusable)):
                    stem = method.replace("/", "_")
                    print(f"cp similarities/{stem}.json similarities/{stem}__p0.json")
            if to_run:
                print("\n# Arms that must actually run:")
                for method, arm in to_run:
                    print(
                        f"python scripts/analysis/math_vs_word.py "
                        f"--method {method!r} --instruction {arm}"
                    )
        broken = broken or to_run

    if not broken:
        print("\nNothing to run.")
    raise SystemExit(1 if broken else 0)

if __name__ == "__main__":
    main()
