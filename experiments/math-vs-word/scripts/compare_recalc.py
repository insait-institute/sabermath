"""Diff a --force_recalc sweep against the snapshot it replaced.

    python3 scripts/compare_recalc.py
    python3 scripts/compare_recalc.py --baseline-dir similarities_baseline

Pairs similarities/<stem>.json against <baseline-dir>/<stem>.json for every
method in scripts/recalc_methods.txt and reports, per method:

  * per-target drift in all three similarity fields (max and mean |delta|),
  * the headline math-vs-word statistic and how far it moved,
  * how many targets FLIPPED that verdict.

The headline number is the one plot_hist.py actually plots: the fraction of
targets where math_score > text_score (plot_hist.py:551). Raw similarity
drift on its own says little - two runs can differ in the 8th decimal on
every target and produce a byte-identical histogram, or agree to 1e-4
everywhere and still flip a handful of near-tied targets. Both are reported
because only the second one changes a published figure.

Stdlib only, deliberately: this must run on a login node in whatever
environment is loaded, not inside one of the eight per-method conda envs.
Same reason check_coverage.py evaluates load_models.py's literals with `ast`
instead of importing it.
"""

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent

SIM_FIELDS = (
    "pr_full_vs_candidates",
    "pr_math_vs_candidates",
    "pr_text_vs_candidates",
)
TEXT_KEY = "pr_text_vs_candidates"
MATH_KEY = "pr_math_vs_candidates"


def load_json(path: Path, attempts: int = 4) -> dict:
    """Read a JSON file that a running job may be rewriting underneath us.

    Retries a truncated/partial read before giving up: the write window is
    milliseconds, so one short sleep is almost always enough.
    """
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25)
    raise AssertionError("unreachable")


def roster(path: Path) -> list[str]:
    """The method ids from recalc_methods.txt (first '|'-separated field)."""
    methods = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        methods.append(line.split("|")[0])
    return methods


def math_greater_rate(payload: dict) -> tuple[int, int]:
    """(targets where math > text, targets counted).

    Ties are excluded rather than counted either way, matching
    plot_hist.py's own handling (it skips them for the models in
    MODELS_WITH_EXPECTED_TIES and refuses to guess for the rest).
    """
    greater = counted = 0
    for row in payload.values():
        text, math = row.get(TEXT_KEY), row.get(MATH_KEY)
        if text is None or math is None or text == math:
            continue
        counted += 1
        greater += math > text
    return greater, counted


def compare(method: str, sim_dir: Path, baseline_dir: Path) -> dict | None:
    stem = method.replace("/", "_")
    new_path, old_path = sim_dir / f"{stem}.json", baseline_dir / f"{stem}.json"

    if not old_path.exists():
        return {"method": method, "state": "no-baseline"}
    if not new_path.exists():
        return {"method": method, "state": "no-result"}

    try:
        old = load_json(old_path)
        new = load_json(new_path)
    except json.JSONDecodeError:
        # sim_embeddings.py rewrites the whole file after every target. It
        # does so atomically now (tmp + os.replace), but a job started
        # before that change still truncates in place, so a read can land
        # mid-write. Not an error worth aborting the whole comparison for.
        return {"method": method, "state": "mid-write"}

    shared = sorted(set(old) & set(new))
    only_old, only_new = sorted(set(old) - set(new)), sorted(set(new) - set(old))

    drift = {f: {"max": 0.0, "sum": 0.0} for f in SIM_FIELDS}
    flips = []
    for tid in shared:
        o, n = old[tid], new[tid]
        for field in SIM_FIELDS:
            ov, nv = o.get(field), n.get(field)
            if ov is None or nv is None:
                continue
            delta = abs(float(nv) - float(ov))
            drift[field]["max"] = max(drift[field]["max"], delta)
            drift[field]["sum"] += delta

        o_verdict = (o.get(MATH_KEY), o.get(TEXT_KEY))
        n_verdict = (n.get(MATH_KEY), n.get(TEXT_KEY))
        if None in o_verdict or None in n_verdict:
            continue
        if (o_verdict[0] > o_verdict[1]) != (n_verdict[0] > n_verdict[1]):
            flips.append(tid)

    old_g, old_n = math_greater_rate(old)
    new_g, new_n = math_greater_rate(new)

    return {
        "method": method,
        "state": "compared",
        "n_old": len(old),
        "n_new": len(new),
        "shared": len(shared),
        "only_old": only_old,
        "only_new": only_new,
        "drift": drift,
        "flips": flips,
        "old_rate": (old_g / old_n if old_n else None, old_g, old_n),
        "new_rate": (new_g / new_n if new_n else None, new_g, new_n),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-dir", type=Path, default=EXP_DIR / "similarities")
    parser.add_argument(
        "--baseline-dir", type=Path, default=EXP_DIR / "similarities_baseline"
    )
    parser.add_argument("--roster", type=Path, default=HERE / "recalc_methods.txt")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Max |delta| treated as numerically identical (default 1e-6).",
    )
    parser.add_argument(
        "--show-flips",
        type=int,
        default=5,
        help="How many flipped target ids to print per method (default 5).",
    )
    args = parser.parse_args()

    results = [compare(m, args.sim_dir, args.baseline_dir) for m in roster(args.roster)]

    changed = 0
    for r in results:
        print(f"\n=== {r['method']}")
        if r["state"] == "no-baseline":
            print(f"  [SKIP] no {args.baseline_dir.name}/ snapshot - nothing to "
                  f"compare against.")
            continue
        if r["state"] == "no-result":
            print("  [SKIP] no similarities/ file - the recalc job has not "
                  "produced output yet.")
            continue
        if r["state"] == "mid-write":
            print("  [SKIP] file is being rewritten right now (the job is "
                  "mid-target) - re-run in a moment.")
            continue

        if r["n_old"] != r["n_new"]:
            print(f"  [!] target count changed: {r['n_old']} -> {r['n_new']}"
                  f"  (a short new file means the job is still running or was "
                  f"preempted)")
        if r["only_old"]:
            print(f"  [!] {len(r['only_old'])} target(s) only in the baseline, "
                  f"e.g. {r['only_old'][:3]}")
        if r["only_new"]:
            print(f"  [!] {len(r['only_new'])} target(s) only in the new run, "
                  f"e.g. {r['only_new'][:3]}")

        worst = max(d["max"] for d in r["drift"].values())
        for field in SIM_FIELDS:
            d = r["drift"][field]
            mean = d["sum"] / r["shared"] if r["shared"] else 0.0
            print(f"  {field:<24} max |d| {d['max']:.3e}   mean |d| {mean:.3e}")

        o_rate, o_g, o_n = r["old_rate"]
        n_rate, n_g, n_n = r["new_rate"]
        if o_rate is not None and n_rate is not None:
            print(f"  math > text               {o_rate:6.2%} ({o_g}/{o_n})"
                  f"  ->  {n_rate:6.2%} ({n_g}/{n_n})"
                  f"   delta {n_rate - o_rate:+.2%}")

        if r["flips"]:
            shown = ", ".join(r["flips"][: args.show_flips])
            more = "" if len(r["flips"]) <= args.show_flips else ", ..."
            print(f"  [FLIP] {len(r['flips'])} target(s) changed the math>text "
                  f"verdict: {shown}{more}")

        if r["flips"] or worst > args.tol or r["only_old"] or r["only_new"]:
            changed += 1
            print(f"  VERDICT: CHANGED")
        else:
            print(f"  VERDICT: identical within {args.tol:g}")

    comparable = [r for r in results if r["state"] == "compared"]
    print(f"\n{'=' * 68}")
    print(f"{len(comparable)}/{len(results)} method(s) compared, "
          f"{changed} changed beyond {args.tol:g}.")
    if len(comparable) < len(results):
        print("Skipped methods are listed above - a missing similarities/ file "
              "usually just means the job has not finished.")
    raise SystemExit(1 if changed else 0)


if __name__ == "__main__":
    main()
