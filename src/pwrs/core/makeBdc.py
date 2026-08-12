# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
from scipy import sparse

from ..corex import FloatArray, MatpowerCase
from ..utils import as_column
from .idx_brch import BR_STATUS, BR_X, F_BUS, SHIFT, T_BUS, TAP
from .idx_bus import BUS_I


def makeBdc_mpc(
    mpc: MatpowerCase,
) -> tuple[sparse.csc_matrix, sparse.csc_matrix, np.ndarray, np.ndarray]:
    """Build the DC power flow ``Bbus`` and ``Bf`` matrices.

    Returns the B matrices and phase shift injection vectors needed for
    a DC power flow. The bus real power injections are related to bus
    voltage angles by::

        P = BBUS * Va + PBUSINJ

    The real power flows at the from end the lines are related to the bus
    voltage angles by::

        Pf = BF * Va + PFINJ

    Does appropriate conversions to p.u.
    Bus numbers must be consecutive beginning at 1 (i.e. internal ordering).

    Parameters
    ----------
    mpc : MatpowerCase
        MATPOWER case struct.

    Returns
    -------
    tuple
        ``(Bbus, Bf, Pbusinj, Pfinj)`` for the DC model.
    """

    return makeBdc_values(mpc["baseMVA"], mpc["bus"], mpc["branch"])


def makeBdc_values(
    baseMVA: float,
    bus: FloatArray,
    branch: FloatArray,
) -> tuple[sparse.csc_matrix, sparse.csc_matrix, np.ndarray, np.ndarray]:
    """Build the DC power flow ``Bbus`` and ``Bf`` matrices.

    Returns the B matrices and phase shift injection vectors needed for
    a DC power flow. The bus real power injections are related to bus
    voltage angles by::

        P = BBUS * Va + PBUSINJ

    The real power flows at the from end the lines are related to the bus
    voltage angles by::

        Pf = BF * Va + PFINJ

    Does appropriate conversions to p.u.
    Bus numbers must be consecutive beginning at 1 (i.e. internal ordering).

    Parameters
    ----------
    baseMVA : float
        System MVA base.
    bus : npt.NDArray[np.float64]
        Bus matrix.
    branch : npt.NDArray[np.float64]
        Branch matrix.

    Returns
    -------
    tuple
        ``(Bbus, Bf, Pbusinj, Pfinj)`` for the DC model.
    """
    bus = np.atleast_2d(np.asarray(bus, dtype=float))
    branch = np.atleast_2d(np.asarray(branch, dtype=float))

    nb = bus.shape[0]
    nl = branch.shape[0]

    if np.any(bus[:, BUS_I] != np.arange(1, nb + 1)):
        raise ValueError(
            "makeBdc: buses must be numbered consecutively in bus matrix; use ext2int() to convert to internal ordering"
        )

    stat = branch[:, BR_STATUS]
    b = stat / branch[:, BR_X]
    tap = np.ones(nl, dtype=float)
    nonzero_tap = np.flatnonzero(branch[:, TAP])
    tap[nonzero_tap] = branch[nonzero_tap, TAP]
    b = b / tap

    f = branch[:, F_BUS].astype(int) - 1
    t = branch[:, T_BUS].astype(int) - 1
    rows = np.r_[np.arange(nl), np.arange(nl)]
    cols = np.r_[f, t]

    Cft = sparse.csc_matrix((np.r_[np.ones(nl), -np.ones(nl)], (rows, cols)), shape=(nl, nb))
    Bf = sparse.csc_matrix((np.r_[b, -b], (rows, cols)), shape=(nl, nb))
    Bbus = Cft.T @ Bf

    Pfinj = b * (-branch[:, SHIFT] * np.pi / 180.0)
    Pbusinj = Cft.T @ as_column(Pfinj)

    return Bbus, Bf, as_column(np.asarray(Pbusinj).reshape(-1)), as_column(Pfinj)


def makeBdc(*args: Any) -> tuple[sparse.csc_matrix, sparse.csc_matrix, np.ndarray, np.ndarray]:
    """Build the DC power flow ``Bbus`` and ``Bf`` matrices.

    Returns the B matrices and phase shift injection vectors needed for
    a DC power flow. The bus real power injections are related to bus
    voltage angles by::

        P = BBUS * Va + PBUSINJ

    The real power flows at the from end the lines are related to the bus
    voltage angles by::

        Pf = BF * Va + PFINJ

    Does appropriate conversions to p.u.
    Bus numbers must be consecutive beginning at 1 (i.e. internal ordering).

    Parameters
    ----------
    *args : tuple
        Positional arguments for one of the supported call forms.

    Returns
    -------
    tuple
        ``(Bbus, Bf, Pbusinj, Pfinj)`` for the DC model.
    """
    if len(args) == 1:
        return makeBdc_mpc(args[0])
    if len(args) == 3:
        return makeBdc_values(args[0], args[1], args[2])
    raise TypeError("makeBdc: expected (mpc) or (baseMVA, bus, branch)")
