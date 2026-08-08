# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


def dSbus_dV(Ybus, V, vcart=0):
    """Compute partial derivatives of bus power injections w.r.t. voltage.

    Parameters
    ----------
    Ybus : array_like or sparse matrix
        Bus admittance matrix.
    V : array_like
        Complex bus voltage vector.
    vcart : int, optional
        Coordinate selector. ``0`` uses polar derivatives with respect to
        voltage angle and magnitude, ``1`` uses cartesian derivatives with
        respect to real and imaginary voltage parts.

    Returns
    -------
    tuple
        ``(dSbus_dV1, dSbus_dV2)`` where the derivatives are with respect
        to ``(Va, Vm)`` in polar mode or ``(Vr, Vi)`` in cartesian mode.
    """
    n = len(V)
    Ibus = Ybus @ V

    if sparse.issparse(Ybus):
        diagV = sparse.diags(V, offsets=0, shape=(n, n), format="csc")
        diagIbus = sparse.diags(Ibus, offsets=0, shape=(n, n), format="csc")
        if not vcart:
            diagVnorm = sparse.diags(V / np.abs(V), offsets=0, shape=(n, n), format="csc")
    else:
        diagV = np.diag(V)
        diagIbus = np.diag(Ibus)
        if not vcart:
            diagVnorm = np.diag(V / np.abs(V))

    if vcart:
        dSbus_dV1 = diagIbus.conjugate() + diagV @ Ybus.conjugate()  # dSbus/dVr
        dSbus_dV2 = 1j * (diagIbus.conjugate() - diagV @ Ybus.conjugate())  # dSbus/dVi
    else:
        dSbus_dV1 = 1j * diagV @ (diagIbus - Ybus @ diagV).conjugate()  # dSbus/dVa
        dSbus_dV2 = diagV @ (Ybus @ diagVnorm).conjugate() + diagIbus.conjugate() @ diagVnorm  # dSbus/dVm

    return dSbus_dV1, dSbus_dV2
