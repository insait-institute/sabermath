import os
import re

import numpy as np


def get_chunks(path: str, prefix: str = "chunk_", suffix: str = ".bin") -> list[str]:
    pattern = re.compile(f"{prefix}([0-9]+){suffix}$")
    files = [f for f in os.listdir(path) if pattern.match(f)]
    files = sorted(files, key=lambda f: int(pattern.match(f).group(1)))
    return files


def get_similarities(
    idx: int,  # row index
    total: int,  # total number of rows
    path: str,  # chunk directory
) -> tuple[np.ndarray, np.ndarray]:  # (bma, jaccard)
    chunk_files = get_chunks(path)
    chunk_count = len(chunk_files)
    chunk_size = total // chunk_count
    chunk_idx = min(idx // chunk_size, chunk_count - 1)
    chunk_name = f"chunk_{chunk_idx}.bin"
    start = total * (idx - chunk_idx * chunk_size)

    chunk_path = os.path.join(path, chunk_name)
    itemsize = np.dtype(np.float16).itemsize

    byte_start = start * itemsize
    byte_len = total * itemsize

    with open(chunk_path, "rb") as f:
        f.seek(2 * byte_start)
        data = f.read(2 * byte_len)

    snapshot = np.frombuffer(data, dtype=np.float16)
    bma = snapshot[0::2]
    sim = snapshot[1::2]
    # scores = bma * sim # snapshot.copy()
    # scores[idx] = -np.inf
    return bma, sim
