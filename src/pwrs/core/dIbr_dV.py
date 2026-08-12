# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from ..corex import ComplexArray, FloatArray, Matrix, as_csc_matrix, complex_matvec


def dIbr_dV(
    branch: FloatArray,
    Yf: Matrix,
    Yt: Matrix,
    V: ComplexArray,
    vcart: int = 0,
) -> tuple[Matrix, Matrix, Matrix, Matrix, ComplexArray, ComplexArray]:
    """Compute partial derivatives of branch currents w.r.t. voltage.

    Parameters
    ----------
    branch : ndarray
        Branch matrix. Accepted for MATLAB interface compatibility.
    Yf, Yt : array_like or sparse matrix
        Branch admittance matrices for the from and to ends.
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
        ``(dIf_dV1, dIf_dV2, dIt_dV1, dIt_dV2, If, It)`` in MATLAB-compatible
        form.
    """
    V = np.asarray(V).reshape(-1)
    nb = len(V)

    if vcart:
        dIf_dV1 = Yf
        dIf_dV2 = 1j * Yf
        dIt_dV1 = Yt
        dIt_dV2 = 1j * Yt
    else:
        Vnorm = V / np.abs(V)
        if sparse.issparse(Yf):
            Yf = as_csc_matrix(Yf)
            Yt = as_csc_matrix(Yt)
            diagV = sparse.diags(V, offsets=0, shape=(nb, nb), format="csc")
            diagVnorm = sparse.diags(Vnorm, offsets=0, shape=(nb, nb), format="csc")
        else:
            Yf = np.asarray(Yf)
            Yt = np.asarray(Yt)
            diagV = np.diag(V)
            diagVnorm = np.diag(Vnorm)
        dIf_dV1 = Yf @ (1j * diagV)
        dIf_dV2 = Yf @ diagVnorm
        dIt_dV1 = Yt @ (1j * diagV)
        dIt_dV2 = Yt @ diagVnorm

    If = complex_matvec(Yf, V)
    It = complex_matvec(Yt, V)

    return dIf_dV1, dIf_dV2, dIt_dV1, dIt_dV2, If, It
