# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


def dImis_dV(Sbus, Ybus, V, vcart=0, *, nargout=None):
    """Compute partial derivatives of current mismatch w.r.t. voltage.

    Parameters
    ----------
    Sbus : array_like
        Complex bus power injections.
    Ybus : array_like or sparse matrix
        Bus admittance matrix.
    V : array_like
        Complex bus voltage vector.
    vcart : int, optional
        Coordinate selector. ``0`` uses polar derivatives with respect to
        voltage angle and magnitude, ``1`` uses cartesian derivatives with
        respect to real and imaginary voltage parts.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    tuple
        ``(dImis_dV1, dImis_dV2)`` in MATLAB-compatible form.
    """
    Sbus = np.asarray(Sbus).reshape(-1)
    V = np.asarray(V).reshape(-1)
    n = V.size

    if vcart:
        diag = np.conj(Sbus / (V**2))
        if sparse.issparse(Ybus):
            Ybus = Ybus.tocsc()
            diagSV2c = sparse.diags(diag, offsets=0, shape=(n, n), format="csc")
        else:
            Ybus = np.asarray(Ybus)
            diagSV2c = np.diag(diag)
        dImis_dV1 = Ybus + diagSV2c
        dImis_dV2 = 1j * (Ybus - diagSV2c)
    else:
        Vm = np.abs(V)
        Ibus = np.conj(Sbus / V)
        if sparse.issparse(Ybus):
            Ybus = Ybus.tocsc()
            diagV = sparse.diags(V, offsets=0, shape=(n, n), format="csc")
            diagIbus = sparse.diags(Ibus, offsets=0, shape=(n, n), format="csc")
            diagIbusVm = sparse.diags(Ibus / Vm, offsets=0, shape=(n, n), format="csc")
            diagVnorm = sparse.diags(V / np.abs(V), offsets=0, shape=(n, n), format="csc")
        else:
            Ybus = np.asarray(Ybus)
            diagV = np.diag(V)
            diagIbus = np.diag(Ibus)
            diagIbusVm = np.diag(Ibus / Vm)
            diagVnorm = np.diag(V / np.abs(V))
        dImis_dV1 = 1j * (Ybus @ diagV - diagIbus)
        dImis_dV2 = Ybus @ diagVnorm + diagIbusVm

    return dImis_dV1, dImis_dV2
