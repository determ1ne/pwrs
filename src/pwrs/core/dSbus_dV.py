# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import cast

import numpy as np
from scipy import sparse

from ..corex import ComplexArray, Matrix, as_csc_matrix, complex_matvec


def dSbus_dV(Ybus: Matrix, V: ComplexArray, vcart: int = 0) -> tuple[Matrix, Matrix]:
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
    Ibus = complex_matvec(Ybus, V)

    if sparse.issparse(Ybus):
        Ybus_sparse = as_csc_matrix(Ybus)
        diagV = sparse.diags(V, offsets=0, shape=(n, n), format="csc")
        diagIbus = sparse.diags(Ibus, offsets=0, shape=(n, n), format="csc")
        diagVnorm = sparse.diags(V / np.abs(V), offsets=0, shape=(n, n), format="csc")
        if vcart:
            dSbus_dV1 = diagIbus.conjugate() + diagV @ Ybus_sparse.conjugate()
            dSbus_dV2 = 1j * (diagIbus.conjugate() - diagV @ Ybus_sparse.conjugate())
        else:
            dSbus_dV1 = 1j * diagV @ (diagIbus - Ybus_sparse @ diagV).conjugate()
            dSbus_dV2 = (
                diagV @ (Ybus_sparse @ diagVnorm).conjugate() + diagIbus.conjugate() @ diagVnorm
            )
        return as_csc_matrix(dSbus_dV1), as_csc_matrix(dSbus_dV2)

    Ybus_dense = np.asarray(Ybus)
    diagV = np.diag(V)
    diagIbus = np.diag(Ibus)
    diagVnorm = np.diag(V / np.abs(V))

    if vcart:
        dSbus_dV1 = diagIbus.conjugate() + diagV @ Ybus_dense.conjugate()
        dSbus_dV2 = 1j * (diagIbus.conjugate() - diagV @ Ybus_dense.conjugate())
    else:
        dSbus_dV1 = 1j * diagV @ (diagIbus - Ybus_dense @ diagV).conjugate()
        dSbus_dV2 = (
            diagV @ (Ybus_dense @ diagVnorm).conjugate() + diagIbus.conjugate() @ diagVnorm
        )

    return cast(tuple[Matrix, Matrix], (dSbus_dV1, dSbus_dV2))
