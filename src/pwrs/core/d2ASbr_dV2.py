# Copyright (c) 2008-2019, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .d2Abr_dV2 import d2Abr_dV2
from .d2Sbr_dV2 import d2Sbr_dV2_full


def _connection_indices(connection) -> np.ndarray:
    if sparse.issparse(connection):
        matrix = sparse.csr_matrix(connection)
        if np.any(matrix.getnnz(axis=1) != 1):
            raise ValueError("d2ASbr_dV2: Cbr must have exactly one nonzero entry per row")
        return np.asarray(matrix.indices[matrix.indptr[:-1]], dtype=int)
    return np.asarray(connection, dtype=int).reshape(-1)


def d2ASbr_dV2(dSbr_dV1, dSbr_dV2, Sbr, Cbr, Ybr, V, mu, vcart=0, nargout=1):
    """Compatibility wrapper for Hessians of squared branch-power magnitudes."""
    connection_indices = _connection_indices(Cbr)
    second_derivative = lambda voltage, multipliers: d2Sbr_dV2_full(
        connection_indices, Ybr, voltage, multipliers, vcart
    )
    return d2Abr_dV2(second_derivative, dSbr_dV1, dSbr_dV2, Sbr, V, mu, nargout)


def d2ASbr_dV2_full(dSbr_dV1, dSbr_dV2, Sbr, Cbr, Ybr, V, mu, vcart=0):
    """Return all four Hessian blocks for squared branch-power magnitudes."""
    return d2ASbr_dV2(dSbr_dV1, dSbr_dV2, Sbr, Cbr, Ybr, V, mu, vcart, nargout=4)
