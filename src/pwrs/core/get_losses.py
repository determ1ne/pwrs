# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .idx_brch import BR_B, BR_R, BR_STATUS, BR_X, F_BUS, SHIFT, T_BUS, TAP
from .idx_bus import BUS_I, VA, VM


def _build_e2i(bus: np.ndarray) -> dict[int, int]:
    i2e = bus[:, BUS_I - 1].astype(int)
    return {ext: idx for idx, ext in enumerate(i2e)}


def _map_e2i(e2i: dict[int, int], bus_numbers: np.ndarray) -> np.ndarray:
    bus_numbers = np.asarray(bus_numbers, dtype=int).reshape(-1)
    return np.fromiter((e2i[int(bus)] for bus in bus_numbers), dtype=int, count=bus_numbers.size)


def get_losses(baseMVA, bus=None, branch=None, *, nargout=None):
    """Compute branch series losses and charging injections.

    It returns complex series losses for each branch, and optionally
    the line charging injections and their voltage derivatives.

    Parameters
    ----------
    baseMVA : float or dict
        System base MVA, or a MATPOWER case struct.
    bus : array_like, optional
        Bus matrix when ``baseMVA`` is not a case struct.
    branch : array_like, optional
        Branch matrix when ``baseMVA`` is not a case struct.
    nargout : int, optional
        MATLAB compatibility flag controlling how many auxiliary outputs are
        returned.

    Returns
    -------
    numpy.ndarray or tuple
        Complex branch loss vector, optionally followed by charging terms and
        derivative dictionaries.
    """
    if isinstance(baseMVA, dict):
        mpc = baseMVA
        baseMVA = mpc["baseMVA"]
        bus = mpc["bus"]
        branch = mpc["branch"]

    baseMVA = float(np.asarray(baseMVA).reshape(-1)[0])
    bus = np.atleast_2d(np.asarray(bus, dtype=float))
    branch = np.atleast_2d(np.asarray(branch, dtype=float))

    e2i = _build_e2i(bus)
    out = np.flatnonzero(branch[:, BR_STATUS - 1] == 0)

    nb = bus.shape[0]
    nl = branch.shape[0]

    V = bus[:, VM - 1] * np.exp(1j * np.pi / 180.0 * bus[:, VA - 1])

    branch_f_idx = _map_e2i(e2i, branch[:, F_BUS - 1])
    branch_t_idx = _map_e2i(e2i, branch[:, T_BUS - 1])
    Cf = sparse.csc_matrix((branch[:, BR_STATUS - 1], (np.arange(nl), branch_f_idx)), shape=(nl, nb))
    Ct = sparse.csc_matrix((branch[:, BR_STATUS - 1], (np.arange(nl), branch_t_idx)), shape=(nl, nb))
    tap = np.ones(nl, dtype=complex)
    xfmr = np.flatnonzero(branch[:, TAP - 1])
    tap[xfmr] = branch[xfmr, TAP - 1]
    tap = tap * np.exp(1j * np.pi / 180.0 * branch[:, SHIFT - 1])
    A = sparse.diags(1 / tap, offsets=0, shape=(nl, nl), format="csc") @ Cf - Ct
    Ysc = 1 / (branch[:, BR_R - 1] - 1j * branch[:, BR_X - 1])
    Vdrop = A @ V
    loss = baseMVA * Ysc * Vdrop * np.conj(Vdrop)

    if nargout is None or nargout <= 1:
        return loss

    Vf = Cf @ V
    Vt = Ct @ V
    fchg = np.real(baseMVA / 2 * branch[:, BR_B - 1] * Vf * np.conj(Vf) / (tap * np.conj(tap)))
    tchg = np.real(baseMVA / 2 * branch[:, BR_B - 1] * Vt * np.conj(Vt))
    fchg[out] = 0
    tchg[out] = 0

    if nargout == 2:
        return loss, fchg + tchg
    if nargout == 3:
        return loss, fchg, tchg

    B = (
        sparse.diags(A @ V, offsets=0, shape=(nl, nl), format="csc")
        @ A.conjugate()
        @ sparse.diags(np.conj(V), offsets=0, shape=(nb, nb), format="csc")
    )
    dYsc = sparse.diags(Ysc, offsets=0, shape=(nl, nl), format="csc")
    dloss_dV = {
        "a": -1j * baseMVA * dYsc @ (B - B.conjugate()),
        "m": baseMVA
        * dYsc
        @ (B + B.conjugate())
        @ sparse.diags(1 / np.abs(V), offsets=0, shape=(nb, nb), format="csc"),
    }

    if nargout == 4:
        return loss, fchg, tchg, dloss_dV

    Bc = sparse.diags(branch[:, BR_B - 1], offsets=0, shape=(nl, nl), format="csc")
    tt = sparse.diags(1 / (tap * np.conj(tap)), offsets=0, shape=(nl, nl), format="csc")
    dchg_dVm = {
        "f": baseMVA * Bc @ tt @ sparse.diags(Cf @ bus[:, VM - 1], offsets=0, shape=(nl, nl), format="csc") @ Cf,
        "t": baseMVA * Bc @ sparse.diags(Ct @ bus[:, VM - 1], offsets=0, shape=(nl, nl), format="csc") @ Ct,
    }
    return loss, fchg, tchg, dloss_dV, dchg_dVm
