#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import json
import re
from pathlib import Path

SHARD_RE = re.compile(r"^(?P<stem>.+)__shard(?P<i>\d+)of(?P<n>\d+)$")
EXPECTED_TARGETS = 969  # RAG4Math/targets_fixed_filtered_latex, train split


def parse_shard(path: Path):
    m = SHARD_RE.match(path.stem)
    if not m:
        raise SystemExit(
            f"{path.name} is not a shard file - expected "
            "<method>[__<arm>]__shard<i>of<n>.json"
        )
    return m.group("stem"), int(m.group("i")), int(m.group("n"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", type=Path, help="Shard result files")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: the shared stem, i.e. the unsuffixed file "
        "the unsharded run would have written).",
    )
    ap.add_argument(
        "--expect",
        type=int,
        default=EXPECTED_TARGETS,
        help=f"Target count to check the union against (default {EXPECTED_TARGETS}). "
        "0 disables the check.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Write even when shards are missing or the union is short.",
    )
    args = ap.parse_args(argv)

    stems, ns, shards = set(), set(), {}
    for path in sorted(args.parts):
        stem, i, n = parse_shard(path)
        stems.add(stem)
        ns.add(n)
        if i in shards:
            raise SystemExit(f"Shard {i} given twice ({shards[i]} and {path})")
        shards[i] = path

    if len(stems) != 1:
        raise SystemExit(f"Shards belong to different runs: {sorted(stems)}")
    if len(ns) != 1:
        raise SystemExit(f"Mixed shard counts: {sorted(ns)}")
    stem, n = stems.pop(), ns.pop()

    print(f"method/arm: {stem}   shards: {len(shards)} of {n}")

    missing = [i for i in range(n) if i not in shards]
    if missing:
        msg = f"missing shard(s) {missing} of {n}"
        if not args.force:
            raise SystemExit(f"[!!] {msg} - pass --force to merge anyway.")
        print(f"[!] {msg} - merging anyway (--force)")

    merged, owner = {}, {}
    for i in sorted(shards):
        path = shards[i]
        with open(path) as fh:
            part = json.load(fh)
        overlap = merged.keys() & part.keys()
        if overlap:
            example = sorted(overlap)[:3]
            raise SystemExit(
                f"[!!] shard{i} overlaps shard{owner[example[0]]} on "
                f"{len(overlap)} target(s), e.g. {example}. Strided shards are "
                "disjoint by construction, so this means the shards were built "
                "against different target sets - do not merge them."
            )
        for k in part:
            owner[k] = i
        merged.update(part)
        print(f"  {path.name}: {len(part)} targets")

    out = args.out or (shards[sorted(shards)[0]].parent / f"{stem}.json")
    total = len(merged)
    print(f"  merged {total} targets -> {out}")

    if args.expect and total != args.expect:
        msg = f"union has {total} targets, expected {args.expect}"
        if not args.force:
            raise SystemExit(
                f"[!!] {msg} - a shard is still running, or one is short. "
                "Pass --force to write it anyway."
            )
        print(f"[!] {msg} - writing anyway (--force)")

    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(merged))
    tmp.replace(out)
    print(f"[+] Wrote {out}")


if __name__ == "__main__":
    main()
