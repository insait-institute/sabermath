from typing import Sequence
from . import _c_bma

_SimilarityMatrix = _c_bma.SimilarityMatrix


class SimilarityMatrix:
    """
    High-level wrapper to the underlying C-level dense similarity matrix.
    """

    def __init__(self, matrix: Sequence[Sequence[float]]):
        try:
            import numpy as np

            if isinstance(matrix, np.ndarray):
                self._matrix = _SimilarityMatrix(matrix.tolist())
                return

        except ImportError:
            pass

        self._matrix = _SimilarityMatrix(matrix)

    def compute_bma(
        self,
        lhs_idxs: list[int] | list[list[int]],
        rhs_idxs: list[int] | list[list[int]],
    ) -> list[list[float]] | list[float] | float:
        """
        Function for computing BMA (Best-Match Average) given indices
        from the Similarity Matrix.
        """

        if not isinstance(lhs_idxs, list) or not isinstance(rhs_idxs, list):
            raise ValueError("Arguments must be lists or lists of lists of indices")
        if len(lhs_idxs) == 0 or len(rhs_idxs) == 0:
            raise ValueError("Argument lists cannot be empty")

        if isinstance(lhs_idxs[0], list) and isinstance(rhs_idxs[0], list):
            return _c_bma.bma_compute(self._matrix, lhs_idxs, rhs_idxs)

        if isinstance(lhs_idxs[0], list):
            return self.compute_bma(rhs_idxs, lhs_idxs)

        if isinstance(rhs_idxs[0], list):
            return _c_bma.bma_compute(self._matrix, [lhs_idxs], rhs_idxs)[0]

        return _c_bma.bma_compute(self._matrix, [lhs_idxs], [rhs_idxs])[0][0]


def compute(
    sim: SimilarityMatrix | Sequence[Sequence[float]],
    lhs_idxs: list[int] | list[list[int]],
    rhs_idxs: list[int] | list[list[int]],
) -> list[list[float]] | list[float] | float:
    if not isinstance(sim, SimilarityMatrix):
        sim = SimilarityMatrix(sim)
    return sim.compute_bma(lhs_idxs, rhs_idxs)
