# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def make_vcorr(DD, pv, nb, nl, f, Zb):
    """Build the PV-bus voltage correction vector for radial PF.

    Mirrors MATPOWER's ``make_vcorr`` helper. It propagates PV-bus
    correction currents through the radial tree to produce a complex voltage
    correction vector for all buses.

    Parameters
    ----------
    DD : array_like
        Complex correction current injections at PV buses.
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
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    numpy.ndarray
        Column vector of complex bus voltage corrections.
    """
    DD = np.asarray(DD, dtype=complex).reshape(-1)
    pv = np.asarray(pv, dtype=int).reshape(-1) - 1
    nb = int(np.asarray(nb).reshape(-1)[0])
    nl = int(np.asarray(nl).reshape(-1)[0])
    f = np.asarray(f, dtype=int).reshape(-1) - 1
    Zb = np.asarray(Zb, dtype=complex).reshape(-1)

    V_corr = np.zeros(nb, dtype=complex)
    I = np.zeros(nb, dtype=complex)
    I[pv] = DD
    for k in range(nl - 1, 0, -1):
        i = f[k]
        I[i] = I[i] + I[k]
    for k in range(1, nl):
        i = f[k]
        V_corr[k] = V_corr[i] - Zb[k] * I[k]
    return V_corr.reshape(-1, 1)
