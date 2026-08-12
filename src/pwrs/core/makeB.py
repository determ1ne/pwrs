# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from ..corex import matrix_imag
from .idx_brch import BR_B, BR_R, SHIFT, TAP
from .idx_bus import BS
from .makeYbus import makeYbus_matrix


def _normalize_makeB_args(baseMVA, bus=None, branch=None, alg=None):
    if branch is None:
        mpc = baseMVA
        if bus is not None:
            alg = bus
        baseMVA = mpc["baseMVA"]
        bus = mpc["bus"]
        branch = mpc["branch"]

    if alg is None:
        raise TypeError("makeB: ALG is required")

    bus = np.atleast_2d(np.asarray(bus, dtype=float))
    branch = np.atleast_2d(np.asarray(branch, dtype=float))

    if not isinstance(alg, str):
        alg_value = alg
        if alg_value == 2:
            alg = "FDXB"
        elif alg_value == 3:
            alg = "FDBX"
    alg = str(alg).upper()
    if alg not in {"FDXB", "FDBX"}:
        raise ValueError(f"makeB: '{alg}' is not a valid value for ALG")
    return baseMVA, bus, branch, alg


def makeB_pair(baseMVA, bus=None, branch=None, alg=None):
    """Return ``(Bp, Bpp)`` with explicit Python semantics."""
    baseMVA, bus, branch, alg = _normalize_makeB_args(baseMVA, bus, branch, alg)

    nb = bus.shape[0]
    nl = branch.shape[0]

    temp_branch = branch.copy()
    temp_bus = bus.copy()
    temp_bus[:, BS] = np.zeros(nb)
    temp_branch[:, BR_B] = np.zeros(nl)
    temp_branch[:, TAP] = np.ones(nl)
    if alg == "FDXB":
        temp_branch[:, BR_R] = np.zeros(nl)
    Bp = -matrix_imag(makeYbus_matrix(baseMVA, temp_bus, temp_branch))

    temp_branch = branch.copy()
    temp_branch[:, SHIFT] = np.zeros(nl)
    if alg == "FDBX":
        temp_branch[:, BR_R] = np.zeros(nl)
    Bpp = -matrix_imag(makeYbus_matrix(baseMVA, bus, temp_branch))
    return Bp, Bpp


def makeB_matrix(baseMVA, bus=None, branch=None, alg=None):
    """Return ``Bp`` only with explicit Python semantics."""
    return makeB_pair(baseMVA, bus, branch, alg)[0]


def makeB(baseMVA, bus=None, branch=None, alg=None, *, nargout=None):
    """Build fast-decoupled power flow ``B'`` and ``B''`` matrices.

    Mirrors MATPOWER's ``makeB`` helper by constructing the susceptance
    matrices used by the FDXB or FDBX fast-decoupled power flow algorithms.

    Parameters
    ----------
    baseMVA : float or dict
        System base MVA, or a MATPOWER case struct.
    bus : array_like or str, optional
        Bus matrix, or algorithm selector when ``baseMVA`` is a case struct.
    branch : array_like, optional
        Branch matrix when ``baseMVA`` is not a case struct.
    alg : str or numeric, optional
        Fast-decoupled variant, ``'FDXB'`` or ``'FDBX'``.
    nargout : int, optional
        MATLAB compatibility flag controlling whether both ``Bp`` and ``Bpp``
        are returned.

    Returns
    -------
    scipy.sparse.spmatrix or tuple
        ``Bp`` alone, or ``(Bp, Bpp)`` when ``nargout > 1``.
    """
    if nargout == 1:
        return makeB_matrix(baseMVA, bus, branch, alg)
    return makeB_pair(baseMVA, bus, branch, alg)
