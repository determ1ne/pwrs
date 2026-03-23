# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


def _row_scale_real_imag(dF_dV, coeff_r, coeff_i):
    """Return ``diag(coeff_r)*real(dF_dV) + diag(coeff_i)*imag(dF_dV)``."""
    if sparse.issparse(dF_dV):
        dF_dV = dF_dV.tocsc(copy=False)
        out = dF_dV
        out.data = coeff_r[out.indices] * np.real(out.data) + coeff_i[out.indices] * np.imag(out.data)
        return out
    return coeff_r[:, None] * np.real(dF_dV) + coeff_i[:, None] * np.imag(dF_dV)


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
    coeff_fr = 2 * np.real(Ff)
    coeff_fi = 2 * np.imag(Ff)
    coeff_tr = 2 * np.real(Ft)
    coeff_ti = 2 * np.imag(Ft)

    dAf_dV1 = _row_scale_real_imag(dFf_dV1, coeff_fr, coeff_fi)
    dAf_dV2 = _row_scale_real_imag(dFf_dV2, coeff_fr, coeff_fi)
    dAt_dV1 = _row_scale_real_imag(dFt_dV1, coeff_tr, coeff_ti)
    dAt_dV2 = _row_scale_real_imag(dFt_dV2, coeff_tr, coeff_ti)

    return dAf_dV1, dAf_dV2, dAt_dV1, dAt_dV2
