# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping

import numpy as np
from scipy import sparse

from ..corex import (
    ArrayLike,
    FloatArray,
    ReadOnlyCaseMapping,
    SparseMatrix,
    as_csc_matrix,
    as_float_matrix,
    as_float_scalar,
)
from .idx_brch import BR_B, BR_R, BR_STATUS, BR_X, F_BUS, SHIFT, T_BUS, TAP
from .idx_bus import BS, BUS_I, GS


def _normalize_makeYbus_args(
    baseMVA: float | ReadOnlyCaseMapping,
    bus: ArrayLike | None = None,
    branch: ArrayLike | None = None,
) -> tuple[float, FloatArray, FloatArray]:
    if branch is None:
        if not isinstance(baseMVA, Mapping):
            raise TypeError("makeYbus: branch is required when baseMVA is numeric")
        mpc = baseMVA
        return (
            as_float_scalar(mpc["baseMVA"], name="baseMVA"),
            as_float_matrix(mpc["bus"], name="bus"),
            as_float_matrix(mpc["branch"], name="branch"),
        )
    if bus is None:
        raise TypeError("makeYbus: bus is required when branch is supplied")
    if not isinstance(baseMVA, (int, float)):
        raise TypeError("makeYbus: baseMVA must be numeric when bus and branch are supplied")
    return float(baseMVA), as_float_matrix(bus, name="bus"), as_float_matrix(branch, name="branch")


def makeYbus_full(
    baseMVA: float | ReadOnlyCaseMapping,
    bus: ArrayLike | None = None,
    branch: ArrayLike | None = None,
) -> tuple[SparseMatrix, SparseMatrix, SparseMatrix]:
    """Return ``(Ybus, Yf, Yt)`` with explicit Python semantics."""
    baseMVA, bus, branch = _normalize_makeYbus_args(baseMVA, bus, branch)

    nb = bus.shape[0]
    nl = branch.shape[0]

    if np.any(bus[:, BUS_I - 1] != np.arange(1, nb + 1)):
        raise ValueError(
            "makeYbus: buses must be numbered consecutively in bus matrix; use ext2int() to convert to internal ordering"
        )

    stat = branch[:, BR_STATUS - 1]
    Ys = stat / (branch[:, BR_R - 1] + 1j * branch[:, BR_X - 1])
    Bc = stat * branch[:, BR_B - 1]
    tap = np.ones(nl, dtype=complex)
    nonzero_tap = np.nonzero(branch[:, TAP - 1])[0]
    tap[nonzero_tap] = branch[nonzero_tap, TAP - 1]
    tap = tap * np.exp(1j * np.pi / 180.0 * branch[:, SHIFT - 1])
    Ytt = Ys + 1j * Bc / 2.0
    Yff = Ytt / (tap * np.conj(tap))
    Yft = -Ys / np.conj(tap)
    Ytf = -Ys / tap

    Ysh = (bus[:, GS - 1] + 1j * bus[:, BS - 1]) / baseMVA

    f = branch[:, F_BUS - 1].astype(int) - 1
    t = branch[:, T_BUS - 1].astype(int) - 1

    rows = np.concatenate([np.arange(nl), np.arange(nl)])
    cols = np.concatenate([f, t])
    Yf = sparse.csc_matrix((np.concatenate([Yff, Yft]), (rows, cols)), shape=(nl, nb))
    Yt = sparse.csc_matrix((np.concatenate([Ytf, Ytt]), (rows, cols)), shape=(nl, nb))

    ybus_rows = np.concatenate([f, f, t, t])
    ybus_cols = np.concatenate([f, t, f, t])
    ybus_data = np.concatenate([Yff, Yft, Ytf, Ytt])
    Ybus = sparse.csc_matrix((ybus_data, (ybus_rows, ybus_cols)), shape=(nb, nb))
    Ybus = Ybus + sparse.diags(Ysh, offsets=0, shape=(nb, nb), format="csc")
    return as_csc_matrix(Ybus), as_csc_matrix(Yf), as_csc_matrix(Yt)


def makeYbus_matrix(
    baseMVA: float | ReadOnlyCaseMapping,
    bus: ArrayLike | None = None,
    branch: ArrayLike | None = None,
) -> SparseMatrix:
    """Return ``Ybus`` only with explicit Python semantics."""
    return makeYbus_full(baseMVA, bus, branch)[0]


def makeYbus(
    baseMVA: float | ReadOnlyCaseMapping,
    bus: ArrayLike | None = None,
    branch: ArrayLike | None = None,
    *,
    nargout: int | None = None,
) -> SparseMatrix | tuple[SparseMatrix, SparseMatrix, SparseMatrix]:
    """Build the bus admittance matrix and branch admittance matrices.

    Parameters
    ----------
    baseMVA : float or dict
        System base MVA, or an MATPOWER case dict containing ``baseMVA``,
        ``bus`` and ``branch``.
    bus : array_like, optional
        MATPOWER bus matrix when ``baseMVA`` is numeric.
    branch : array_like, optional
        MATPOWER branch matrix when ``baseMVA`` is numeric.
    nargout : int, optional
        MATLAB-compatibility output selector. If ``nargout == 1``, only
        ``Ybus`` is returned.

    Returns
    -------
    scipy.sparse.csc_matrix or tuple
        Returns ``Ybus`` alone when ``nargout == 1``. Otherwise returns
        ``(Ybus, Yf, Yt)``, where ``Yf`` and ``Yt`` map bus voltages to branch
        currents injected at the from and to ends of each branch.

    Notes
    -----
    Inputs may be supplied either as an MATPOWER case dict or as separate
    ``baseMVA``, ``bus`` and ``branch`` arguments. Bus numbers must use
    internal ordering, i.e. be consecutive starting at 1.

    See Also
    --------
    makeJac, makeSbus, ext2int
    """
    if nargout == 1:
        return makeYbus_matrix(baseMVA, bus, branch)
    return makeYbus_full(baseMVA, bus, branch)
