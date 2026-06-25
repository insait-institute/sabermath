"""
This file contains implementations of some metrics related to
the evaluation of Embedding Models on Mathematical data.

References:
[1] https://yulab-smu.top/biomedical-knowledge-mining-book/semantic-similarity-overview.html
"""

from typing import Sequence
import numpy as np


def informational_content(p, eps=1e-12):
    p = max(p, eps)
    return -np.log(p)


def lin_similarity(p_1, p_2, p_mica, eps=1e-12) -> np.float64:
    """
    Lin Similarity from [1]

    Parameters
    ----------
    p_1, p_2 :  float
                probabilities of each node
    p_mica :    float
                probability of the Most Informative Common Ancestor (MICA)
                in the tree case this is the Lowest Common Ancestor (LCA)

    Returns
    ----------
    np.float64
        The Lin Similarity
    """
    IC_1 = informational_content(p_1)
    IC_2 = informational_content(p_2)
    IC_mica = informational_content(p_mica)
    return (2 * IC_mica) / (IC_1 + IC_2 + eps)
