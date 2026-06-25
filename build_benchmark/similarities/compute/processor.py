import os
import multiprocessing
import psutil
import traceback
from typing import Sequence, Any

import fastbma
from datasets import load_dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm

from jaccard import jaccard_similarity, tokenize
from l2 import get_l2_cache_groups
from tree import Tree


class SimilarityProcessor:
    def __init__(
        self,
        hf_path: str,
        tree: Tree,
        similarity_matrix: np.ndarray,
        *,
        datasets: list[str] | None = None,
        tags_column: str = "tags",
        ideas_column: str = "idea",
        parallelize: bool = True,
    ):
        if similarity_matrix.ndim != 2:
            raise ValueError("Expected a 2D matrix")

        self._sim_npy = similarity_matrix

        print("[~] Loading databank..")
        ds = load_dataset(hf_path)

        if datasets is None:
            datasets = ds.keys()

        ds = concatenate_datasets([ds[name] for name in datasets])

        self._row_count = ds.num_rows
        self._column_count = self._row_count

        # tags to idxs
        print("[~] Processing tags...")
        self._tags: list[list[int]] = [
            [tree.node_index_by_path(tag) for tag in row[tags_column]] for row in ds
        ]

        # tokenize ideas
        print("[~] Processing core ideas...")
        self._ideas: list[set[str]] = [set(tokenize(row[ideas_column])) for row in ds]

        del ds

        self._multiprocess = parallelize

    def _combine_metrics(self, bma: np.ndarray, jaccard: np.ndarray):
        return bma * jaccard

    def _process_single_chunk(
        self,
        start_row: int,
        end_row: int,
        save_path: str,
        similarity_matrix: fastbma.SimilarityMatrix,
        progress_queue: multiprocessing.Queue,
    ) -> None:
        scores = np.empty(
            (end_row - start_row, 2 * self._column_count), dtype=np.float16
        )

        for idx in range(start_row, end_row):
            # Compute BMA
            lhs_idxs = self._tags[idx]
            rhs_idxs = self._tags
            bma = fastbma.compute(similarity_matrix, lhs_idxs, rhs_idxs)
            bma = np.array(bma, dtype=np.float16)

            # Compute Jaccard
            current_idea_toks = self._ideas[idx]
            jaccard = [
                jaccard_similarity(current_idea_toks, rhs_toks)
                for rhs_toks in self._ideas
            ]
            jaccard = np.array(jaccard, dtype=np.float16)

            # score = bma * jaccard
            # scores[idx - start_row] = score
            scores[idx - start_row, 0::2] = bma
            scores[idx - start_row, 1::2] = jaccard

            if progress_queue is not None:
                progress_queue.put(1)

        # scores = (np.vstack(scores, dtype=np.float16)
        #           if len(scores) > 0 else np.array([], dtype=np.float16))
        scores.tofile(save_path)

    def _process_chunk_range(
        self,
        row_intervals: list[tuple[int, int]],
        initial_chunk_number: int,
        progress_queue: multiprocessing.Queue,
        chunk_file_prefix: str = "../data/chunk_",
        chunk_file_suffix: str = ".bin",
        pin_to_cpu: int | None = None,
    ) -> None:
        similarity_matrix = fastbma.SimilarityMatrix(self._sim_npy)
        if pin_to_cpu is not None:
            proc = psutil.Process(os.getpid())
            proc.cpu_affinity([pin_to_cpu])

        for i, (start, end) in enumerate(row_intervals, start=initial_chunk_number):
            self._process_single_chunk(
                start,
                end,
                f"{chunk_file_prefix}{i}{chunk_file_suffix}",
                similarity_matrix,
                progress_queue,
            )

    def compute(
        self,
        output_path: str = "../data/",
        chunk_count: int = 100,
        show_progress_bar: bool = True,
        max_parallel: int | None = None,
    ) -> None:
        os.makedirs(output_path, exist_ok=True)
        save_prefix = os.path.join(output_path, "chunk_")

        chunk_size = self._row_count // chunk_count
        chunks = [
            (i * chunk_size, (i + 1) * chunk_size) for i in range(chunk_count - 1)
        ]
        chunks.append(((chunk_count - 1) * chunk_size, self._row_count))

        if self._multiprocess:
            l2_cache_groups = get_l2_cache_groups()
            cpu_ids = [group[0] for group in l2_cache_groups]
            if max_parallel is not None and max_parallel < len(cpu_ids):
                cpu_ids = cpu_ids[:max_parallel]
            job_count = len(cpu_ids)
            print(f"[+] Splitting into {job_count} processes.")

            if job_count == 0:
                raise RuntimeError("No available CPUs")

            job_size = chunk_count // job_count
            job_intervals = [
                chunks[i * job_size : (i + 1) * job_size] for i in range(job_count - 1)
            ]
            job_intervals.append(chunks[(job_count - 1) * job_size :])

            manager = multiprocessing.Manager()
            prog_q = manager.Queue(maxsize=0)

            with multiprocessing.Pool(processes=job_count) as pool:
                current_chunk = 0
                jobs = []
                for cid, intervals in zip(cpu_ids, job_intervals):
                    job = pool.apply_async(
                        self._process_chunk_range,
                        (intervals, current_chunk, prog_q),
                        {"chunk_file_prefix": save_prefix, "pin_to_cpu": cid},
                    )
                    jobs.append(job)
                    current_chunk += len(intervals)

                done = 0

                if show_progress_bar:
                    with tqdm(total=self._row_count) as pbar:
                        while done < self._row_count:
                            try:
                                prog_q.get()
                                done += 1
                                pbar.update()
                            except:
                                break

                for job in jobs:
                    try:
                        job.get()
                    except Exception as e:
                        traceback.print_exception(type(e), e, e.__traceback__)

        else:
            similarity_matrix = fastbma.SimilarityMatrix(self._sim_npy)

            if show_progress_bar:
                with tqdm(total=self._row_count) as pbar:
                    done = 0
                    for i, (start, end) in enumerate(chunks):
                        save_path = f"{save_prefix}{i}.bin"
                        self._process_single_chunk(
                            start, end, save_path, similarity_matrix, None
                        )
                        done += end - start
                        pbar.update(end - start)
            else:
                for i, (start, end) in enumerate(chunks):
                    save_path = f"{save_prefix}{i}.bin"
                    self._process_single_chunk(start, end, save_path, None)
