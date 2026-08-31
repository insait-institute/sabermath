"""Coverage check for the math-vs-word experiment.

    python src/sabermath/analysis/math_vs_word/check_coverage.py
    python src/sabermath/analysis/math_vs_word/check_coverage.py --emit-commands

Answers one question: for every method calc_sims.py can run, is there a
similarities/<method>.json holding a COMPLETE result?

Completeness is the point. calc_embedding_sims() checkpoints after every
single target and resumes from len(similarities_dict) on the next run, so a
preempted job leaves behind a file that exists, parses, and is short. `ls`
cannot tell that apart from a finished run; this can. Any method that is
missing, short, or carries a null similarity is reported, and with
--emit-commands the exact calc_sims.py invocation to fix it is printed.

Reads ALLOWED_MODELS straight out of load_models.py rather than hardcoding a
roster, so a model added there is picked up here with no edit.
"""

import argparse
import ast
import json
from pathlib import Path
from . import SIMILARITIES_DIR

HERE = Path(__file__).resolve().parent
EXPECTED_TARGETS = 969  # RAG4Math/targets_fixed_filtered_latex, train split
SIM_FIELDS = (
    "pr_full_vs_candidates",
    "pr_math_vs_candidates",
    "pr_text_vs_candidates",
)
NON_EMBEDDING_METHODS = ["jaccard", "approach0", "tf-idf", "bm25"]


def envelope_for(method: str) -> dict | None:
    """The model's canonical vendor input envelope, or None if it cannot be
    resolved (a non-embedding method, or a model with no registry key).

    Reported for information only. It used to decide whether an instructed
    p0 arm needed its own run, back when the default arm here was
    prompt-free and p0 was not: for a model with an empty envelope the two
    were identical, for the rest they were different runs.

    Since 2026-08-26 the default arm IS p0 under the canonical protocol
    (load_models.get_scores_kwargs(method) and
    get_scores_kwargs(method, "p0") return the same thing by construction),
    so <method>.json and <method>__p0.json are the same run for EVERY
    model, empty envelope or not - see load_models.py's header. A p0 arm is
    therefore always copyable from the default file.
    """
    import sys

    sys.path.insert(0, str(HERE))
    try:
        from .load_models import get_scores_kwargs

        return dict(get_scores_kwargs(method, "p0"))
    except Exception:
        return None


def literal_env(path: Path, env: dict | None = None) -> dict:
    """Evaluate a module's top-level assignments that reduce to literals,
    into `env`. Anything that does not (a function call, an import-bound
    name we have no value for) is skipped rather than raising."""
    env = {} if env is None else env
    tree = ast.parse(path.read_text())

    def evaluate(node):
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.Subscript):
            return evaluate(node.value)[ast.literal_eval(node.slice)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return evaluate(node.left) + evaluate(node.right)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [evaluate(e) for e in node.elts]
        if isinstance(node, ast.Dict):
            return {
                ast.literal_eval(k): evaluate(v)
                for k, v in zip(node.keys, node.values)
            }
        return ast.literal_eval(node)

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                env[target.id] = evaluate(node.value)
            except (ValueError, KeyError, TypeError, AttributeError):
                pass  # not a plain literal - irrelevant to the roster
    return env


def allowed_models(path: Path, extra_sources: list[Path]) -> list[str]:
    """Evaluate load_models.py's module-level string lists WITHOUT importing
    it - the import pulls in torch, vllm and sabermath's processors, none of
    which a coverage check should need (and which are not installed in every
    environment this runs from).

    ALLOWED_MODELS is built as ALLOWED_MODELS + ADDITIONAL_MODELS, and
    ADDITIONAL_MODELS references RADER_BIENCODER_MODELS["rader-14b"] and
    friends, which load_models.py imports from scripts/run_experiments.py, so
    that module's literals are evaluated into the same namespace first.
    Without it ADDITIONAL_MODELS silently fails to resolve and the roster
    comes back as just the original 17 - a coverage check that quietly
    ignores half the models is worse than no check.
    """
    env: dict[str, object] = {}
    for source in extra_sources:
        if source.exists():
            literal_env(source, env)
    literal_env(path, env)

    models = env.get("ALLOWED_MODELS")
    if not models:
        raise SystemExit(f"Could not read ALLOWED_MODELS from {path}")
    return list(models)


def inspect(method: str, sim_dir: Path, arm: str | None = None) -> tuple[str, str]:
    """(status, detail) for one method. status in ok/missing/short/null/bad."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-dir", type=Path, default=SIMILARITIES_DIR)
    parser.add_argument("--load-models", type=Path, default=HERE / "load_models.py")
    parser.add_argument(
        "--extra-source",
        type=Path,
        nargs="*",
        default=[HERE.parent.parent / "scripts" / "run_experiments.py"],
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
    args = parser.parse_args()

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
                f"python calc_sims.py --config_file config.yaml "
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
                # p0 == the default arm since 2026-08-26 (see
                # envelope_for), so it is always a copy, never a run.
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
                        f"python calc_sims.py --config_file config.yaml "
                        f"--method {method!r} --instruction {arm}"
                    )
        broken = broken or to_run

    if not broken:
        print("\nNothing to run.")
    raise SystemExit(1 if broken else 0)


if __name__ == "__main__":
    main()
