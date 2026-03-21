# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


def dAbr_dV(dFf_dV1, dFf_dV2, dFt_dV1, dFt_dV2, Ff, Ft, *, nargout=None):
    """Compute derivatives of squared flow magnitudes w.r.t. voltage.

    Parameters
    ----------
    dFf_dV1, dFf_dV2, dFt_dV1, dFt_dV2 : array_like or sparse matrix
        Flow sensitivities with respect to the two voltage-coordinate
        components.
    Ff, Ft : array_like
        Complex or real flows at the from and to ends.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    tuple
        ``(dAf_dV1, dAf_dV2, dAt_dV1, dAt_dV2)``.
    """
    Ff = np.asarray(Ff).reshape(-1)
    Ft = np.asarray(Ft).reshape(-1)
    nl = len(Ff)

    dAf_dFfr = sparse.diags(2 * np.real(Ff), offsets=0, shape=(nl, nl), format="csc")
    dAf_dFfi = sparse.diags(2 * np.imag(Ff), offsets=0, shape=(nl, nl), format="csc")
    dAt_dFtr = sparse.diags(2 * np.real(Ft), offsets=0, shape=(nl, nl), format="csc")
    dAt_dFti = sparse.diags(2 * np.imag(Ft), offsets=0, shape=(nl, nl), format="csc")

    dAf_dV1 = dAf_dFfr @ np.real(dFf_dV1) + dAf_dFfi @ np.imag(dFf_dV1)
    dAf_dV2 = dAf_dFfr @ np.real(dFf_dV2) + dAf_dFfi @ np.imag(dFf_dV2)
    dAt_dV1 = dAt_dFtr @ np.real(dFt_dV1) + dAt_dFti @ np.imag(dFt_dV1)
    dAt_dV2 = dAt_dFtr @ np.real(dFt_dV2) + dAt_dFti @ np.imag(dFt_dV2)

    return dAf_dV1, dAf_dV2, dAt_dV1, dAt_dV2
