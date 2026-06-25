import argparse
from dataclasses import dataclass
import math
from multiprocessing import Pool
from typing import Literal, Iterator

import numpy as np
from tqdm import tqdm

from similarities import get_similarities
from utils import (
    load_data,
    get_domains,
    DOMAIN_LOWER_TO_STYLIZED,
)


@dataclass(frozen=True)
class WorkerInfo:
    name: Literal["bma", "jaccard"]
    row_count: int
    idxs: list[int]
    path: str
    begin: int | None = 0
    count: int | None = None

    def with_range(self, begin: int, count: int) -> "WorkerInfo":
        return WorkerInfo(
            name=self.name,
            row_count=self.row_count,
            idxs=self.idxs,
            path=self.path,
            begin=begin,
            count=count,
        )


def make_iterator(
    name: Literal["bma", "jaccard"],
    total_count: int,  # num_rows of the ENTIRE dataset
    idxs: list[int],
    path: str,
    begin: int = 0,
    count: int | None = None,
) -> Iterator[np.ndarray]:
    if begin < 0:
        raise ValueError("'begin' must be non-negative")

    if count is None:
        count = len(idxs) - begin
    elif count < 0:
        raise ValueError("'count' must be non-negative")

    if name not in ("bma", "jaccard"):
        raise ValueError("'name' must be either 'bma' or 'jaccard'")

    for i in idxs[begin : begin + count]:
        bma, jaccard = get_similarities(i, total_count, path)
        yield bma if name == "bma" else jaccard


def _process_chunk(
    info: WorkerInfo,
    low: float,
    high: float,
    bucket_count: int,
    need_finite_total: bool = False,
) -> tuple[int, int, int, np.ndarray]:
    if high <= low:
        raise ValueError("'high' must be greater than 'low'")

    scale = bucket_count / (high - low)

    buckets = np.zeros(bucket_count, dtype=np.int64)
    under = 0
    in_range = 0
    finite_total = 0

    it = make_iterator(
        name=info.name,
        total_count=info.row_count,
        idxs=info.idxs,
        path=info.path,
        begin=info.begin,
        count=info.count,
    )

    for arr in it:
        arr = arr.astype(np.float64, copy=False)
        arr = arr[np.isfinite(arr)]
        n = int(arr.size)

        if need_finite_total:
            finite_total += n

        if n == 0:
            continue

        under += int(np.count_nonzero(arr < low))

        arr_in = arr[(arr >= low) & (arr <= high)]
        m = int(arr_in.size)
        if m == 0:
            continue

        in_range += m

        idx = ((arr_in - low) * scale).astype(np.int64, copy=False)
        idx[idx == bucket_count] = bucket_count - 1
        buckets += np.bincount(idx, minlength=bucket_count).astype(np.int64, copy=False)

    return finite_total, under, in_range, buckets


def _process_chunk_star(
    args: tuple[WorkerInfo, float, float, int, bool],
) -> tuple[int, int, int, np.ndarray]:
    return _process_chunk(*args)


def _scan_parallel(
    worker_info: WorkerInfo,
    low: float,
    high: float,
    bucket_count: int,
    target_rank: int | None,
    workers: int = 1,
    chunk_size: int = 256,
    show_progress_bar: bool = True,
) -> tuple[int, int, int, np.ndarray]:
    if workers <= 0:
        raise ValueError("'workers' must be >= 1")

    need_finite_total = target_rank is None
    total_items = len(worker_info.idxs)

    jobs: list[tuple[WorkerInfo, float, float, int, bool]] = []
    for begin in range(0, total_items, chunk_size):
        count = min(chunk_size, total_items - begin)

        info = worker_info.with_range(begin, count)

        jobs.append((info, low, high, bucket_count, need_finite_total))

    finite_total = 0
    under = 0
    in_range = 0
    buckets = np.zeros(bucket_count, dtype=np.int64)

    with Pool(processes=workers) as pool:
        results = pool.imap_unordered(_process_chunk_star, jobs)

        if show_progress_bar:
            results = tqdm(results, total=len(jobs), desc=f"Scan ({worker_info.name})")

        for part_finite_total, part_under, part_in_range, part_buckets in results:
            finite_total += part_finite_total
            under += part_under
            in_range += part_in_range
            buckets += part_buckets

    return finite_total, under, in_range, buckets


def compute_threshold(
    info: WorkerInfo,
    percentile: float,
    low: float = 0.0,
    high: float = 1.0,
    bucket_count: int = 1024,
    rounds: int = 4,
    current_round: int = 1,
    workers: int = 1,
    show_progress_bar: bool = True,
    # internal (computed on first run)
    target_rank: int | None = None,
    total_values: int | None = None,
) -> tuple[float, float]:
    """
    Multi-pass histogram zoom to estimate the global percentile threshold.
    """

    if not (0.0 <= percentile <= 1.0):
        raise ValueError("percentile must be in (0, 1)")
    if bucket_count <= 1:
        raise ValueError("bucket_count must be > 1")
    if rounds < 0:
        raise ValueError("rounds must be >= 0")
    if not (low <= high):
        raise ValueError("low must be <= high")
    if workers < 1:
        raise ValueError("there must be at least 1 worker")

    if low == high:
        return low, 0.0

    if rounds == 0:
        return (low + high) / 2.0, high - low

    print(
        f"[~] Round {current_round}: Computing histogram with range [{low}, {high}]..."
    )

    finite_total, under, in_range, buckets = _scan_parallel(
        worker_info=info,
        low=low,
        high=high,
        bucket_count=bucket_count,
        target_rank=target_rank,
        workers=workers,
        show_progress_bar=show_progress_bar,
    )

    if target_rank is None:
        total_values = finite_total
        if total_values == 0:
            raise ValueError("No finite values found in the data.")
        target_rank = int(math.ceil(percentile * total_values))

    if in_range == 0:
        raise ValueError("No values found within [low, high].")

    cum = under + np.cumsum(buckets)
    b = int(np.searchsorted(cum, target_rank, side="left"))

    width = (high - low) / bucket_count
    new_low = low + b * width
    new_high = new_low + width

    return compute_threshold(
        info=info,
        percentile=percentile,
        low=new_low,
        high=new_high,
        bucket_count=bucket_count,
        rounds=rounds - 1,
        current_round=current_round + 1,
        show_progress_bar=show_progress_bar,
        workers=workers,
        target_rank=target_rank,
        total_values=total_values,
    )


def percentile_float(value: str) -> float:
    x = float(value)
    if not (0 <= x <= 1):
        raise argparse.ArgumentTypeError(f"{value} is not in the interval [0, 1]")
    return x


def positive_int(value):
    x = int(value)
    if x <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be a positive integer")
    return x


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "databank",
        type=str,
        help="HF Path to the problem databank",
    )

    parser.add_argument(
        "--domain",
        type=str.lower,
        choices=list(DOMAIN_LOWER_TO_STYLIZED.keys()) + ["all"],
        default="all",
        required=False,
        help="Limit problems to a domain",
    )

    parser.add_argument(
        "--percentile",
        type=percentile_float,
        default=0.99,
        required=False,
        help="Percentile threshold to compute [0, 1]",
    )

    parser.add_argument(
        "--similarities",
        type=str,
        default="../data/sim",
        required=False,
        help="Location of the precomputed similarities",
    )

    parser.add_argument(
        "--no-bma",
        action="store_true",
        help="Disable BMA threshold computation",
    )

    parser.add_argument(
        "--no-jaccard",
        action="store_true",
        help="Disable Jaccard threshold computation",
    )

    parser.add_argument(
        "--workers",
        type=positive_int,
        default=1,
        required=False,
        help="Number of worker processes",
    )

    parser.add_argument(
        "--rounds",
        type=positive_int,
        default=4,
        required=False,
        help="Number of zoom rounds",
    )

    parser.add_argument(
        "--no-progress-bars",
        action="store_true",
        help="Disable progress bars",
    )

    args = parser.parse_args()

    if args.no_bma and args.no_jaccard:
        print("[-] Cannot disable both BMA and Jaccard threshold computations.")
        return

    print("[~] Loading databank...")
    ds = load_data(args.databank)

    showp = not args.no_progress_bars
    path = args.similarities
    domain = args.domain

    if domain == "all":
        domain = "Not specified."
        idxs = list(range(ds.num_rows))
    else:
        domain = DOMAIN_LOWER_TO_STYLIZED.get(domain)
        print(f"[~] Filtering problems with domain: {domain}.")
        idxs = [
            i
            for i in tqdm(range(ds.num_rows), disable=not showp)
            if domain in get_domains(ds, i)
        ]
        # idxs = get_domain_idxs(ds, domain, show_progress_bar=showp)

    count = len(idxs)
    print(f"[+] Number of valid problems: {count}")

    bma_threshold = None
    jaccard_threshold = None

    if not args.no_bma:
        print("[~] Computing BMA threshold...")
        bma_info = WorkerInfo("bma", ds.num_rows, idxs, path)
        bma_threshold, bma_rng = compute_threshold(
            bma_info,
            args.percentile,
            rounds=args.rounds,
            workers=args.workers,
            show_progress_bar=showp,
        )
        print(f"[+] BMA Threshold: {bma_threshold} (range: {bma_rng})")

    if not args.no_jaccard:
        print("[~] Computing Jaccard threshold...")
        jaccard_info = WorkerInfo("jaccard", ds.num_rows, idxs, path)
        jaccard_threshold, j_rng = compute_threshold(
            jaccard_info,
            args.percentile,
            rounds=args.rounds,
            workers=args.workers,
            show_progress_bar=showp,
        )
        print(f"[+] Jaccard Threshold: {jaccard_threshold} (range: {j_rng})")

    print(f"\n========================================================================")
    print(f"Percentile: {args.percentile}")
    print(f"Domain: {domain}")
    print(f"Number of problems: {count}")
    if bma_threshold is not None:
        print(f"BMA Threshold: {bma_threshold} (range: {bma_rng})")
    if jaccard_threshold is not None:
        print(f"Jaccard Threshold: {jaccard_threshold} (range: {j_rng})")
    print(f"========================================================================\n")


if __name__ == "__main__":
    main()
