# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def make_zpv(pv, nb, nl, f, Zb, Yd):
    """Build the PV sensitivity matrix for radial PF correction.

    Mirrors MATPOWER's ``make_zpv`` helper. It computes the matrix relating
    PV-bus reactive corrections to voltage changes for the radial
    backward/forward sweep methods.

    Parameters
    ----------
    pv : array_like
        One-based PV bus indices.
    nb : int
        Number of buses.
    nl : int
        Number of branches.
    f : array_like
        One-based parent-bus indices for each branch.
    Zb : array_like
        Branch series impedances.
    Yd : array_like
        Bus shunt admittances.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    numpy.ndarray
        Complex PV sensitivity matrix.
    """
    pv = np.asarray(pv, dtype=int).reshape(-1) - 1
    nb = int(np.asarray(nb).reshape(-1)[0])
    nl = int(np.asarray(nl).reshape(-1)[0])
    f = np.asarray(f, dtype=int).reshape(-1) - 1
    Zb = np.asarray(Zb, dtype=complex).reshape(-1)
    Yd = np.asarray(Yd, dtype=complex).reshape(-1)

    npv = len(pv)
    Zpv = np.zeros((npv, npv), dtype=complex)
    Ye = Yd.copy()
    D = np.zeros(nl, dtype=complex)
    for k in range(nl - 1, 0, -1):
        D[k] = 1.0 / (1.0 + Zb[k] * Ye[k])
        i = f[k]
        Ye[i] = Ye[i] + D[k] * Ye[k]
    for ipv in range(npv):
        V = np.zeros(nb, dtype=complex)
        Je = np.zeros(nb, dtype=complex)
        Je[pv[ipv]] = -1.0
        for k in range(nl - 1, 0, -1):
            i = f[k]
            Je[i] = Je[i] + Je[k]
        for k in range(1, nl):
            i = f[k]
            V[k] = D[k] * (V[i] - Zb[k] * Je[k])
        Zpv[:, ipv] = V[pv]
    return Zpv
