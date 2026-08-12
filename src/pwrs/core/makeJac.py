# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .bustypes import bustypes
from .dSbus_dV import dSbus_dV
from .idx_bus import BUS_TYPE, PV, REF, VA, VM
from .idx_gen import GEN_BUS, GEN_STATUS, VG
from .makeYbus import makeYbus_full


def makeJac_full(baseMVA, bus=None, branch=None, gen=None, fullJac=None):
    """Return ``(J, Ybus, Yf, Yt)`` with explicit Python semantics."""
    if gen is None:
        mpc = baseMVA
        if bus is not None:
            fullJac = bus
        else:
            fullJac = 0
        baseMVA = mpc["baseMVA"]
        bus = mpc["bus"]
        branch = mpc["branch"]
        gen = mpc["gen"]
    elif fullJac is None:
        fullJac = 0

    bus = np.atleast_2d(np.asarray(bus, dtype=float))
    branch = np.atleast_2d(np.asarray(branch, dtype=float))
    gen = np.atleast_2d(np.asarray(gen, dtype=float))
    fullJac = int(np.asarray(fullJac).reshape(-1)[0]) if np.asarray(fullJac).size else 0

    Ybus, Yf, Yt = makeYbus_full(baseMVA, bus, branch)

    V = bus[:, VM] * np.exp(1j * np.pi / 180.0 * bus[:, VA])

    on = np.flatnonzero(gen[:, GEN_STATUS] > 0)
    gbus = gen[on, GEN_BUS].astype(int)
    k = np.flatnonzero((bus[gbus - 1, BUS_TYPE] == PV) | (bus[gbus - 1, BUS_TYPE] == REF))
    if k.size:
        gidx = gbus[k] - 1
        V[gidx] = gen[on[k], VG] / np.abs(V[gidx]) * V[gidx]

    dSbus_dVa, dSbus_dVm = dSbus_dV(Ybus, V)
    if fullJac:
        j11 = np.real(_dense(dSbus_dVa))
        j12 = np.real(_dense(dSbus_dVm))
        j21 = np.imag(_dense(dSbus_dVa))
        j22 = np.imag(_dense(dSbus_dVm))
        J = np.block([[j11, j12], [j21, j22]])
    else:
        ref, pv, pq = bustypes(bus, gen)
        pvpq = np.r_[np.asarray(pv, dtype=int).reshape(-1), np.asarray(pq, dtype=int).reshape(-1)]
        pq = np.asarray(pq, dtype=int).reshape(-1)
        pvpq0 = pvpq - 1
        pq0 = pq - 1

        j11 = np.real(_slice_matrix(dSbus_dVa, pvpq0, pvpq0))
        j12 = np.real(_slice_matrix(dSbus_dVm, pvpq0, pq0))
        j21 = np.imag(_slice_matrix(dSbus_dVa, pq0, pvpq0))
        j22 = np.imag(_slice_matrix(dSbus_dVm, pq0, pq0))
        if sparse.issparse(dSbus_dVa) or sparse.issparse(dSbus_dVm):
            J = sparse.vstack([sparse.hstack([j11, j12]), sparse.hstack([j21, j22])], format="csc")
        else:
            J = np.block([[_dense(j11), _dense(j12)], [_dense(j21), _dense(j22)]])
    return J, Ybus, Yf, Yt


def makeJac_matrix(baseMVA, bus=None, branch=None, gen=None, fullJac=None):
    """Return ``J`` only with explicit Python semantics."""
    return makeJac_full(baseMVA, bus, branch, gen, fullJac)[0]


def makeJac(baseMVA, bus=None, branch=None, gen=None, fullJac=None, *, nargout=None):
    """Form the power flow Jacobian.

    Parameters
    ----------
    baseMVA : float or dict
        System power base, or an MATPOWER case dict in the one-argument
        form.
    bus, branch, gen : ndarray, optional
        MATPOWER matrices in internal ordering.
    fullJac : int or bool, optional
        If true, return the full Jacobian of all bus injections with
        respect to all voltage angles and magnitudes. Otherwise return the
        reduced Newton power flow Jacobian.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    tuple or matrix
        Returns ``(J, Ybus, Yf, Yt)`` or the leading subset requested by
        ``nargout``.
    """
    outputs = makeJac_full(baseMVA, bus, branch, gen, fullJac)
    if nargout == 1:
        return outputs[0]
    if nargout == 2:
        return outputs[:2]
    if nargout == 3:
        return outputs[:3]
    return outputs


def _slice_matrix(matrix, rows, cols):
    if sparse.issparse(matrix):
        return matrix[np.asarray(rows, dtype=int)[:, None], np.asarray(cols, dtype=int)].tocsc()
    return np.asarray(matrix)[np.ix_(np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))]


def _dense(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)
