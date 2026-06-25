import os

import datasets
import numpy as np
from tqdm import tqdm


DOMAIN_TO_TAGS = {
    "Algebra": ["Algebra"],
    "Number Theory": ["NumberTheory"],
    "Geometry": ["Geometry"],
    "Calculus and Analysis": ["CalculusandAnalysis"],
    "Combinatorics": [
        "ProbabilityandStatistics",
        "DiscreteMathematics",
        "RecreationalMathematics",
    ],
}


TAG_TO_DOMAIN = {
    "NumberTheory": "Number Theory",
    "Geometry": "Geometry",
    "Algebra": "Algebra",
    "CalculusandAnalysis": "Calculus and Analysis",
    "DiscreteMathematics": "Combinatorics",
    "ProbabilityandStatistics": "Combinatorics",
    "RecreationalMathematics": "Combinatorics",
}


DOMAIN_LOWER_TO_STYLIZED = {
    "number theory": "Number Theory",
    "nt": "Number Theory",
    "algebra": "Algebra",
    "a": "Algebra",
    "alg": "Algebra",
    "geometry": "Geometry",
    "g": "Geometry",
    "geo": "Geometry",
    "calculus and analysis": "Calculus and Analysis",
    "calculus": "Calculus and Analysis",
    "analysis": "Calculus and Analysis",
    "calc": "Calculus and Analysis",
    "combinatorics": "Combinatorics",
    "c": "Combinatorics",
    "comb": "Combinatorics",
}


CACHE_NAME_TO_DOMAIN = {
    ".ntheory.npy": "Number Theory",
    ".geomtry.npy": "Geometry",
    ".algebra.npy": "Algebra",
    ".combitc.npy": "Combinatorics",
    ".calculs.npy": "Calculus and Analysis",
}


DOMAIN_TO_CACHE_NAME = {
    "Number Theory": ".ntheory.npy",
    "Geometry": ".geomtry.npy",
    "Algebra": ".algebra.npy",
    "Combinatorics": ".combitc.npy",
    "Calculus and Analysis": ".calculs.npy",
}


def load_data(path: str) -> datasets.Dataset:
    ds = datasets.load_dataset(path)
    return datasets.concatenate_datasets(
        [
            ds["numeric"],
            ds["proofs"],
        ]
    )


def get_domains(ds: datasets.Dataset, idx: int) -> list[str]:
    tags = [tag.split("/")[2] for tag in ds[idx]["tags"]]
    domains = set()
    for tag in tags:
        domain = TAG_TO_DOMAIN.get(tag)
        if domain:
            domains.add(domain)
    return list(domains)


def get_domain_idxs(
    ds: datasets.Dataset,
    domain: str,
    *,
    show_progress_bar: bool = True,
    cache_dir=".cache",
    check_cache: bool = True,
    save_cache_if_miss: bool = True,
) -> list[int]:
    if domain not in DOMAIN_TO_CACHE_NAME:
        raise ValueError("Invalid domain.")

    cache_filename = DOMAIN_TO_CACHE_NAME.get(domain)

    if not cache_filename.endswith(".npy"):
        cache_filename += ".npy"

    cache_path = os.path.join(cache_dir, cache_filename)

    if check_cache:
        if os.path.isfile(cache_path):
            return list(np.load(cache_path))

    domain_idxs = []
    for i in tqdm(range(len(ds)), disable=not show_progress_bar):
        if domain in get_domains(ds, i):
            domain_idxs.append(i)

    if save_cache_if_miss:
        os.makedirs(cache_dir, exist_ok=True)
        arr = np.array(domain_idxs)
        np.save(cache_path, arr)

    return domain_idxs
